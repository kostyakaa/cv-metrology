"""Поле стола, мозаика и карта высот.

В окне камеры
    c подключение на видимой угловой метке -> k проверка осей и привязка
    r обход четырёх углов и запись рамки -> t измерение плоскости, если требуется
    p скан всего поля змейкой -> v просмотр или повторное открытие мозаики
    WASD ручной ход, 1/2/3 шаг 0.1/1/10 мм
    b локальный скан высот вокруг текущего XY на сетке --grid x --grid
    g наведение на выбранную точку карты высот

В окне мозаики
    колёсико зум, правая кнопка панорама, левая кнопка выбор XY
    g наведение на выбранный XY, q возврат в окно камеры

Координаты мозаики указаны в масштабе командных мм относительно домашней метки.
--trial запускает отдельную диагностику у одной метки. Переход к рабочему режиму
выполняется повторным запуском скрипта без --trial.

Чтобы снова работать с сохранённой мозаикой, добавьте к рабочей команде запуска
    --open mosaics/<сеанс>/fused
Порядок в открытой сессии (домашняя точка по метке, концевиков нет):
    камеру на любую метку -> c -> k (оси + привязка) -> h (проверка: к домашней метке)
    -> v (мозаика) -> клик -> g.
g едет только внутри зоны, где станок уже был. Если с момента съёмки мозаики менялись
$3/$100/$101 - скрипт это поймает при подключении и мозаику отключит.
--view mosaics/<сеанс> - только посмотреть, без камеры и станка.

При p между остановками записываются оптические смещения и статусы GRBL с временем получения.
После возврата на метку открывается уточнённая мозаика из <сеанс>/fused.
Для старых сеансов: python scripts/rebuild_mosaic.py mosaics/<сеанс>
Для сравнения позиций, перекрытий и меток: python scripts/mosaic_report.py mosaics/<сеанс>/fused
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'z'), str(ROOT/'scripts')]
from common.camera import capture_mode, open_camera, read_frame, verify_focus
from common.grbl import GrblError
from common.markers import detect
from common.mosaic import Mosaic, pixel_to_machine, target_displacement
from common.registration import navigation_quality
from common.odometry import reduced, motion_leg
from rebuild_mosaic import rebuild
from measure_height import HeightTool, MapWindow, add_height_args
from survey_markers import (Abort, WindowUI, add_survey_args, find_calibration, JOG_KEYS, JOG_STEPS,
                            LAYOUT_SETTINGS)


def load_mosaic(folder, K=None, D=None):
    """Папка сеанса скана (или путь к mosaic.json) -> (folder, meta). С K/D - снята ли той же калибровкой."""
    folder = Path(folder)
    if folder.is_file():
        folder = folder.parent
    try:
        meta = json.loads((folder/'mosaic.json').read_text(encoding='utf-8'))
        for name in ('mosaic.png', 'coverage.png'):
            if not (folder/name).exists():
                raise FileNotFoundError(name)
    except (OSError, ValueError) as e:
        raise Abort(f'{folder}: не читается как сеанс скана ({e})')
    for key in ('polygon', 'ppm', 'records', 'home_id', 'camera_matrix', 'dist_coeffs'):
        if key not in meta:
            raise Abort(f'{folder}/mosaic.json: нет "{key}" - чужой или битый формат')
    if K is not None and not np.allclose(meta['camera_matrix'], np.asarray(K, float), rtol=0, atol=1e-6):
        raise Abort(f'мозаика снята другой калибровкой ({meta.get("calibration")}) - масштаб пиксель->мм не тот, '
                    f'координаты не совпадут. Откройте с той калибровкой или пересканируйте (p)')
    if D is not None and not np.allclose(meta['dist_coeffs'], np.asarray(D, float), rtol=0, atol=1e-6):
        raise Abort(f'у мозаики другие коэффициенты дисторсии ({meta.get("calibration")}) - откройте с той '
                    f'калибровкой или пересканируйте (p)')
    return folder, meta


class PlaneScanner(HeightTool):
    motion_samples = None
    trial_origin = None

    def move(self, dx, dy, feed=None, manual=False, check=True):
        if self.args.trial:
            if self.trial_origin is None:
                raise Abort('Проба: сначала подключение и проверка настроек')
            offset = self.position() + [dx, dy] - self.trial_origin
            if not np.isfinite(offset).all() or np.max(np.abs(offset)) > 8:
                raise Abort('Проба: цель вне ±8 мм от положения подключения')
            feed = min(feed or self.args.feed, 120)
        return super().move(dx, dy, feed, manual, check)

    def tick(self, frame=None):
        if self.motion_samples is None:
            return super().tick(frame)
        frame = read_frame(self.cap) if frame is None else frame
        received = time.monotonic()
        # The moving preview only needs display and Esc. Marker detection is performed at stops.
        if self.ui.show(frame) == 27:
            return True
        if received-self.motion_samples[-1][0] >= self.motion_interval:
            self.motion_samples.append((received, reduced(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))
            # Use the most recent normal jog status; do not add serial queries in the stop loop.
            p = self.grbl.reported_position
            if p is not None and hasattr(self.grbl, 'last_status_time'):
                self.motion_status.append(dict(frame_s=received-self.motion_start,
                    status_s=self.grbl.last_status_time-self.motion_start,
                    position=(np.asarray(p[:2])-self.motion_home).tolist()))
            if len(self.motion_samples) >= 64:
                self.motion_samples = self.motion_samples[::2]
                self.motion_interval *= 2
        return False

    mosaic_folder = None   # папка открытой/снятой мозаики; None - мозаики нет
    mosaic_meta = None
    mosaic_home_id = 0     # ID домашней метки, относительно которой координаты мозаики

    def open_mosaic(self, folder):
        """--open: загрузить сохранённый скан. v - окно, клик/g ведут станок после привязки c -> k."""
        self.mosaic_folder, self.mosaic_meta = load_mosaic(folder, self.K, self.D)
        self.mosaic_home_id = int(self.mosaic_meta['home_id'])
        meta = self.mosaic_meta
        if not meta.get('complete'):
            self.say(f'ВНИМАНИЕ: скан не доведён до конца ({len(meta["records"])} кадров) - часть поля пустая')
        if meta.get('focus') is not None and self.focus is not None and int(meta['focus']) != int(self.focus):
            self.say(f'ВНИМАНИЕ: скан снят при фокусе {meta["focus"]}, сейчас {self.focus} - живой кадр другого '
                     f'масштаба; на координаты мозаики это не влияет')
        self.say(f'мозаика открыта: {self.mosaic_folder.name} - {len(meta["records"])} кадров, {meta["ppm"]:g} px/мм, '
                 f'дом ID{self.mosaic_home_id}')
        if meta.get('trial'):
            self.say('v — просмотр короткой пробы. Для наведения по всему полю сделайте полный скан p.')
        else:
            self.say('v — окно мозаики (клик/g). g доступна после привязки на угловой метке (c, k)')

    def mosaic_anchor(self):
        """Домашняя метка скана в координатах станка сейчас (после c + k на любой метке). None - привязки нет."""
        if self.mosaic_folder is None:
            return None
        return self.expected().get(self.mosaic_home_id)

    def check_mosaic_settings(self, cur):
        """$3/$100/$101 против записанных в мозаике на момент съёмки; при расхождении мозаика отключается."""
        if self.mosaic_folder is None:
            return
        saved = (self.mosaic_meta or {}).get('grbl_settings')
        if not saved:
            self.say('в мозаике не записаны $3/$100/$101 на момент съёмки - считаю, что они не менялись')
            return
        changed = [n for n in LAYOUT_SETTINGS if str(n) in saved and cur.get(n) != saved.get(str(n))]
        if changed:
            self.mosaic_folder = None
            raise Abort(f'с съёмки мозаики изменились {", ".join(f"${n}" for n in changed)} - её мм команды не '
                        f'совпадают с текущими. Мозаика отключена; пересканируйте p')

    def connect(self):
        if self.grbl is None:
            if self.require_marker()['id'] not in range(4):
                raise Abort('Перед подключением нужна одна из меток ID0–3')
            self.seen_marker = True
        super().connect()
        try:
            cur = self.grbl.settings()
        except GrblError:
            self.trial_origin = None
            raise
        if self.args.trial:
            self.trial_origin = None
            if (not self.layout or any(cur.get(n) is None or cur[n] != self.layout_settings.get(n)
                                       or cur[n] != self.calib_settings.get(n) for n in LAYOUT_SETTINGS)):
                raise Abort('Проба: настройки GRBL расходятся с калибровкой или раскладкой')
            self.grbl.require_idle()
            # Pressing c again must not reset the travel limit.
            self.trial_origin = self.visited[0].copy()
        self.check_mosaic_settings(cur)

    def capture_color(self):
        deadline=time.monotonic()+self.args.scan_settle
        while time.monotonic()<deadline:
            if self.tick():raise Abort('Capture cancelled')
        samples=[]
        for _ in range(self.args.scan_frames):
            frame=read_frame(self.cap)
            if self.tick(frame):raise Abort('Capture cancelled')
            samples.append(frame.astype(np.float32))
        return np.mean(samples,axis=0).astype(np.uint8)

    def scan_plane(self):
        self.require_grbl()
        if self.A is None:
            raise Abort('Сначала проверка осей k на видимой метке')
        exp = self.expected()
        if set(exp) != set(range(4)):
            raise Abort('Для полного поля нужна раскладка четырёх угловых меток. Выполните r')
        try:
            settings = {str(n): v for n, v in self.grbl.settings().items() if n in LAYOUT_SETTINGS}
        except GrblError as exc:
            raise Abort(f'До съёмки не удалось сверить настройки GRBL: {exc}') from exc
        if any(settings.get(str(n)) != self.layout_settings.get(n) or str(n) not in settings for n in LAYOUT_SETTINGS):
            raise Abort('Настройки GRBL не совпадают с раскладкой меток')
        start = self.position()
        plane, why = self.table_plane_here(start)
        if plane is None:
            raise Abort(f'{why}. Измерьте плоскость клавишей t')
        polygon = cv2.convexHull(np.array(list(exp.values()),np.float32)).reshape(-1,2)
        home_id = int(self.table_plane['home_id'])
        home = np.array(exp[home_id])
        lo, hi = polygon.min(axis=0), polygon.max(axis=0)
        # Conservative spacing from central valid image area in command coordinates.
        H = pixel_to_machine(self.K,self.V,*plane,start)
        w,h = self.size
        corners = np.array([[[.25*w,.25*h],[.75*w,.25*h],[.75*w,.75*h],[.25*w,.75*h]]],np.float64)
        foot = cv2.perspectiveTransform(corners,H)[0]
        step = min(self.args.tile_step, float(np.min(np.ptp(foot,axis=0)))*.5)
        if step < 1:
            raise Abort('Invalid camera footprint')
        axes = [np.linspace(a,b,max(2,int(np.ceil((b-a)/step))+1)) for a,b in zip(lo,hi)]
        targets = [np.array([x,y]) for i,y in enumerate(axes[1])
                   for x in (axes[0] if i%2==0 else axes[0][::-1])
                   if cv2.pointPolygonTest(polygon,(float(x),float(y)),False)>=0]
        marker = self.require_marker()
        if marker['id'] not in exp:
            raise Abort('Для старта нужна одна из четырёх угловых меток')
        if self.args.trial:
            if self.last_centered != marker['id'] or np.linalg.norm(start-exp[marker['id']]) > .5:
                raise Abort('Для пробы выполните k и оставьте камеру на этой метке')
            k = int(np.argmin(np.linalg.norm(polygon-start,axis=1)))
            u,v = polygon[[(k+1)%4,(k-1)%4]]-polygon[k]
            if min(np.linalg.norm(u),np.linalg.norm(v)) < 6:
                raise Abort('Слишком маленькая раскладка для пробы 6 мм')
            u,v = 6*u/np.linalg.norm(u),6*v/np.linalg.norm(v)
            targets = [start,start+u,start+u+v,start+v,start]
            polygon = cv2.convexHull(np.array(targets[:4],np.float32)).reshape(-1,2)
            step = 6.
            self.say('Проба: смещения XY от старта ' + str([np.round(p-start,2).tolist() for p in targets]))
            if np.max(np.abs(np.asarray(targets)-self.trial_origin)) > 8:
                raise Abort('Маршрут выходит за ±8 мм от подключения')
        if not targets or len(targets)>1000:
            raise Abort('Too many tiles')
        # Validate ALL endpoints, including the return, before beginning.
        for p in targets+[start]:
            self.check_target(p)
        if not self.confirm(f'Scan {len(targets)} stops, step <= {step:.1f} command mm, inside marker polygon; clear entire XY path'):
            return
        folder = ROOT/'mosaics'/time.strftime('%Y%m%d-%H%M%S')
        folder.mkdir(parents=True,exist_ok=False)
        mosaic = Mosaic(polygon-home,self.args.ppm)
        meta = dict(version=1, units='command_mm_relative_to_home_marker', home_id=home_id,
                    head_z_mm=self.args.head_z_mm, profile_id=self.profile_id,
                    camera_mode=getattr(self.args, "camera_mode", None),
                    home_session=home.tolist(), polygon=(polygon-home).tolist(), ppm=self.args.ppm,
                    camera_matrix=self.K.tolist(), dist_coeffs=self.D.tolist(), focus=self.focus,
                    calibration=self.calib_name, model=self.calib_model, blend='central-source',
                    grbl_settings=settings, records=[], complete=False, trial=self.args.trial,
                    marker_positions={str(k):(np.asarray(v)-home).tolist() for k,v in exp.items()},
                    marker_size_mm=self.args.marker_size)
        def save():
            meta['pair_errors'] = mosaic.errors
            meta['coverage_fraction'] = float(np.count_nonzero((mosaic.weight>0)&(mosaic.region>0))/max(1,np.count_nonzero(mosaic.region)))
            for name, image in [('mosaic',mosaic.best),('coverage',(mosaic.weight>0).astype(np.uint8)*255)]:
                tmp = folder/f'{name}.tmp.png'
                if not cv2.imwrite(str(tmp),image):
                    raise OSError(f'Cannot save {tmp}')
                tmp.replace(folder/f'{name}.png')
            tmp = folder/'mosaic.tmp.json'
            tmp.write_text(json.dumps(meta,indent=2,allow_nan=False),encoding='utf-8')
            tmp.replace(folder/'mosaic.json')
        try:
            marker = self.require_marker(marker['id'])
            delta = targets[0]-self.position()
            if np.linalg.norm(delta)>self.args.probe*2:
                self.verify_direction(marker['id'],delta)
            start_observation = start + np.linalg.solve(self.A, self.target - marker['center'])
            meta['start_marker_id'] = int(marker['id'])
            meta['start_marker_position'] = (start_observation-home).tolist()
            previous_small = None
            for i,p in enumerate(targets):
                samples = None
                if previous_small is not None:
                    self.motion_start = time.monotonic()
                    self.motion_home = home
                    self.motion_samples = [(self.motion_start, previous_small)]
                    self.motion_interval = .05
                    self.motion_status = []
                try:
                    self.move(*(p-self.position()),feed=min(self.args.feed,600))
                finally:
                    samples, self.motion_samples = self.motion_samples, None
                stopped = time.monotonic()
                raw = self.capture_color()
                captured = time.monotonic()
                pos = self.position()
                if not np.isfinite(pos).all() or np.linalg.norm(pos-p) > .05:
                    raise Abort(f'GRBL после остановки сообщает {pos} вместо цели {p}')
                plane,why = self.table_plane_here(pos)
                if plane is None:
                    raise Abort(why)
                H = pixel_to_machine(self.K,self.V,*plane,pos-home)
                filename = f'frame_{i:04d}.png'
                if not cv2.imwrite(str(folder/filename),raw):
                    raise Abort(f'Cannot save {filename}')
                record = dict(file=filename,position=(pos-home).tolist(),target=(p-home).tolist(),H=H.tolist())
                previous_small = reduced(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY))
                if samples:
                    samples.append((captured, previous_small))
                    self.motion_status.append(dict(status_s=stopped-self.motion_start, position=record['position']))
                    flow = motion_leg(samples, meta['records'][-1], record, self.K, self.D, self.calib_model,
                                      self.motion_status, min(self.args.feed,120 if self.args.trial else 600))
                    intervals = np.diff([t for t, _ in samples])
                    flow['preview_dt_median_s'] = float(np.median(intervals))
                    flow['preview_dt_max_s'] = float(intervals.max())
                    flow['status_samples'] = self.motion_status
                    record['motion_from_previous'] = flow
                meta['records'].append(record)
                if self.args.trial:
                    sight = self.require_marker(marker['id'])
                    displacement = np.linalg.solve(self.A, sight['center']-marker['center'])
                    discrepancy = displacement-(pos-start)
                    if np.isfinite(discrepancy).all():
                        record['trial_marker_error_mm'] = discrepancy.tolist()
                    if 'trial_marker_error_mm' not in record or np.linalg.norm(discrepancy) > .5:
                        raise Abort(f'Проба: движение по метке расходится с GRBL на {discrepancy.round(3)} мм')
                mosaic.add(self.rect(raw),self.rect.valid,H,pos-home,raw=raw,K=self.K,D=self.D,model=self.calib_model)
                save()
                self.say(f'Tile {i+1}/{len(targets)}, coverage {meta["coverage_fraction"]:.1%}')
            self.move(*(start-self.position()),feed=min(self.args.feed,600))
            _,closure = self.center_on(marker['id'])
            meta['return_to_start_mm'] = (closure-start_observation).tolist()
            meta['end_marker_position'] = (closure-home).tolist()
            meta['closure_command_mm'] = (closure-np.array(exp[marker['id']])).tolist()
            meta['complete'] = True
        except (Abort, GrblError, RuntimeError, OSError, cv2.error) as exc:
            meta['stop_reason'] = str(exc)
            raise
        finally:
            # On failure stay stopped; never blindly return after loss of position.
            save()
        self.mosaic_folder = folder
        self.mosaic_meta = meta
        self.mosaic_home_id = int(meta['home_id'])
        self.say('Скан сохранён. Совмещаю остановки по камере, меткам и координатам GRBL...')
        try:
            self.mosaic_folder, self.mosaic_meta = rebuild(folder, folder/'fused')
        except (ValueError, OSError, cv2.error, np.linalg.LinAlgError) as exc:
            self.say(f'Не удалось уточнить мозаику: {exc}. Доступен исходный скан {folder}')
        self.say(f'Мозаика: {self.mosaic_folder}')
        self.show_mosaic()

    def show_mosaic(self):
        if self.mosaic_folder is None:
            raise Abort('мозаики нет: p - снять, --open <сеанс> - открыть сохранённую')
        inspection_only = self.args.trial or self.mosaic_meta.get('trial')
        quality = navigation_quality(self.mosaic_meta)
        if self.mosaic_meta.get('head_z_mm') != getattr(self.args, 'head_z_mm', None):
            quality['reasons'].append('изменилось положение головки Z')
            quality['usable'] = False
        if self.mosaic_meta.get('profile_id') and self.mosaic_meta['profile_id'] != self.profile_id:
            quality['reasons'].append('изменился профиль камеры')
            quality['usable'] = False
        if not inspection_only and not quality['usable']:
            self.say('Мозаика открыта для просмотра. ' + ', '.join(quality['reasons']))
        view(self.mosaic_folder, None if inspection_only or not quality['usable'] else self)

    def refine_mosaic_target(self, image, coverage, uv, ppm):
        """Verify a stopped jog against the mosaic and apply at most two bounded corrections."""
        meta = self.mosaic_meta or {}
        if meta.get('focus') != self.focus or self.mosaic_anchor() is None:
            self.say('Доводка недоступна при другом фокусе или без привязки')
            return
        # Sample a square at the same physical pitch from both images.
        ppm_match = 16.
        r = round(6.*ppm_match)
        side = 2*r+1
        factor = ppm_match/ppm
        to_patch = np.array([[factor,0,r-factor*uv[0]],[0,factor,r-factor*uv[1]]])
        reference = cv2.warpAffine(image,to_patch,(side,side),flags=cv2.INTER_LINEAR)
        valid = cv2.warpAffine(coverage,to_patch,(side,side),flags=cv2.INTER_NEAREST)>0
        previous = None
        for attempt in range(3):
            pos = self.position()
            plane, why = self.table_plane_here(pos)
            if plane is None:
                self.say(f'Доводка по мозаике недоступна. {why}')
                return
            local = pixel_to_machine(self.K,self.V,*plane,[0.,0.])
            grid = np.array([[ppm_match,0,r],[0,ppm_match,r],[0,0,1.]])
            fresh = self.rect(self.capture_color())
            live = cv2.warpPerspective(fresh,grid@local,(side,side))
            delta = target_displacement(reference,live,valid,ppm_match)
            if delta is None:
                self.say('Позиция по мозаике не подтверждена. Недостаточно однозначной текстуры.')
                return
            error = np.linalg.norm(delta)
            self.say(f'Остаток по мозаике X{delta[0]:+.3f} Y{delta[1]:+.3f} мм, модуль {error:.3f} мм')
            if error <= .07:
                return
            if error > 1.2 or (previous is not None and error >= previous*.85):
                self.say('Доводка остановлена. Смещение велико или остаток не уменьшается.')
                return
            if attempt == 2:
                self.say('Достигнут предел двух поправок')
                return
            target = pos+delta
            bc = self.backlash_comp
            self.check_target(target)
            self.check_target(target-bc)
            self.move(*(delta-bc),feed=min(self.args.slow_feed,120))
            self.move(bc,bc,feed=min(self.args.slow_feed,120))
            previous = error

    def scan_focus_z(self):
        from focus_height import acquire
        self.require_grbl()
        acquire(self)

    def scan_heights(self):
        """Use the existing multi-view scan around the camera's current position."""
        self.scan_here()
        if self.map is None:
            self.map = MapWindow('Локальная карта высот')
            self.show_map()

    def poll_height_map(self):
        if self.map is None:
            return
        try:
            visible = cv2.getWindowProperty(self.map.name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = -1  # Backends can throw after the map is closed with its close button.
        if visible < 1:
            self.map = None
            self.sel = None
            return
        click = self.map.take_click()
        if click is not None:
            self.handle_click(click)


def view(folder, tool=None):
    folder, meta = load_mosaic(folder)
    image=cv2.imread(str(folder/'mosaic.png'))
    valid=cv2.imread(str(folder/'coverage.png'),0)
    owners = None
    scale_grid = (meta.get('registration') or {}).get('scale_grid')
    scale_field = np.asarray(scale_grid['values'],np.float32) if scale_grid else None
    if meta.get('registration'):
        owners = cv2.imread(str(folder/meta['tile_index']),cv2.IMREAD_UNCHANGED)
        if (owners is None or owners.dtype != np.uint16 or owners.shape != valid.shape
                or np.any((owners>0) != (valid>0)) or owners.max() > len(meta['records'])):
            raise Abort('Повреждена карта исходных кадров мозаики')
    origin=np.min(meta['polygon'],axis=0)
    scale=min(1.,1200/image.shape[1],800/image.shape[0])
    viewport=[min(1200,image.shape[1]),min(800,image.shape[0])]
    offset=np.zeros(2,dtype=float)
    drag=[]
    selected=[]
    area_corners=[]
    name='Plane mosaic (click: coordinates, a: ROI corner, g: move, q: close)'
    cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE)
    def click(event,x,y,flags,param):
        nonlocal scale,offset
        mouse=np.array([x,y],float)
        if event==cv2.EVENT_MOUSEWHEEL:
            anchor=(mouse-offset)/scale
            scale=float(np.clip(scale*(1.25 if flags>0 else .8),.05,12))
            offset=mouse-anchor*scale
            return
        if event==cv2.EVENT_RBUTTONDOWN:drag[:]=[mouse,offset.copy()]
        elif event==cv2.EVENT_RBUTTONUP:drag.clear()
        elif event==cv2.EVENT_MOUSEMOVE and drag:offset=drag[1]+mouse-drag[0]
        if event!=cv2.EVENT_LBUTTONDOWN:return
        u,v=np.floor((mouse-offset)/scale).astype(int)
        select_pixel(u, v)

    def select_pixel(u, v):
        if not(0<=v<valid.shape[0] and 0<=u<valid.shape[1]) or not valid[v,u]:return
        xy=origin+np.array([u,v])/meta['ppm']
        command_xy = xy
        if owners is not None:
            record = meta['records'][int(owners[v,u])-1]
            offset_mm = np.asarray(record['offset_mm'])
            command_xy = xy-offset_mm
            if scale_field is not None:
                grid_xy = (xy-scale_grid['origin'])/scale_grid['spacing']
                maps = grid_xy.astype(np.float32).reshape(1,1,2)
                local_scale = float(cv2.remap(scale_field,maps,None,cv2.INTER_LINEAR)[0,0])
                if abs(local_scale) <= .04:
                    # Undo the local change in projection scale about the source camera.
                    centre = np.asarray(record['position'])
                    command_xy = centre+(command_xy-centre)/(1+local_scale)
        selected[:] = [xy, command_xy, (u,v)]
        print(f'XY on plane: {xy.round(3)} mm | home-relative GRBL XY: {command_xy.round(3)} mm')
    cv2.setMouseCallback(name,click)
    try:
        while True:
            M=np.array([[scale,0,offset[0]],[0,scale,offset[1]]])
            shown=cv2.warpAffine(image,M,tuple(viewport),flags=cv2.INTER_NEAREST if scale>=1 else cv2.INTER_AREA)
            if selected:
                at=((selected[0]-origin)*meta['ppm']*scale+offset).round().astype(int)
                cv2.drawMarker(shown,tuple(at),(0,0,255),cv2.MARKER_CROSS,21,1)
            anchor = None if tool is None else tool.mosaic_anchor()
            bound = 'view only' if tool is None else ('bound: g moves machine' if anchor is not None
                                                      else 'NOT bound: c,k on a marker first')
            cv2.putText(shown,f'{scale*100:.0f}% | wheel zoom, RMB pan, LMB select | {bound}', (8,22),
                        cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
            cv2.imshow(name,shown)
            key=cv2.waitKey(30)&255
            if key in (27,ord('q')) or cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1:break
            if key == ord('a') and tool is not None and selected:
                area_corners.append(selected[0].copy())
                if len(area_corners) == 1:
                    print("Первая вершина области. Выберите противоположную и нажмите a.")
                else:
                    first, second = area_corners
                    area_corners.clear()
                    dims = np.abs(second-first)
                    if dims.min() < 1:
                        print("Слишком малая область. Выберите углы заново.")
                        continue
                    centre = (first+second)/2
                    uv = np.rint((centre-origin)*meta['ppm']).astype(int)
                    if not valid[uv[1], uv[0]]:
                        print("Центр области не покрыт мозаикой")
                        continue
                    # Use the same owner/scale conversion as a normal click, at full mosaic resolution.
                    select_pixel(*uv)
                    tool.args.area_mm = dims.tolist()
                    print(f"Область {dims.round(2)} мм, запас {tool.args.area_margin:g} мм. "
                          "g — к центру, q — закрыть мозаику, b — скан.")
            if key==ord('g') and tool is not None and selected:
                if anchor is None:
                    print('мозаика не привязана к станку: камеру на любую метку -> c -> k (-> h), снова v и g')
                    continue
                target=selected[1]+anchor
                try:
                    tool.check_target(target)
                    if tool.confirm(f'Camera to mosaic XY {selected[0].round(3).tolist()} '
                                    f'(machine X{target[0]:.2f} Y{target[1]:.2f})?'):
                        bc=tool.backlash_comp
                        tool.check_target(target-bc)
                        tool.move(*(target-tool.position()-bc),feed=min(tool.args.feed,600))
                        tool.move(bc,bc,feed=tool.args.slow_feed,check=False)
                        p=tool.position()
                        print(f'Положение GRBL после основного хода X{p[0]:.3f} Y{p[1]:.3f}')
                        tool.refine_mosaic_target(image, valid, selected[2], meta['ppm'])
                except (Abort,GrblError) as e:
                    if tool.grbl:tool.grbl.cancel()
                    print(f'СТОП: {e}')
    finally:
        cv2.destroyWindow(name)


def preflight(args, names, ports):
    """Read-only inventory. Never instantiate VideoCapture, Grbl or a serial port."""
    errors = []
    calib,path = find_calibration(args.camera,(args.width,args.height),args.calib)
    try:
        frame = json.loads(args.frame_file.read_text(encoding='utf-8'))
        plane = json.loads(args.plane_file.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        print(f'Не удалось прочитать геометрию сеанса: {exc}')
        return False
    if calib is None or path.name != plane['calib']:
        errors.append('Калибровка кадра и плоскости несовместима')
    else:
        mode = calib['mode']
        print(f"{path.name}, {args.width}x{args.height}, focus {mode.get('focus')}, RMS {calib['rms']:.3f} px")
        if not (calib['machine_check'] or {}).get('motion') or mode.get('focus') is None:
            errors.append('В калибровке нет хода станка или значения фокуса')
        if frame.get('focus') != mode.get('focus') or frame.get('frame_size') != calib['image_size']:
            errors.append('Режим камеры при обходе меток отличается от калибровки')
        settings = mode.get('grbl_settings',{})
        print(f'Настройки в файлах: {settings}')
        if not all(str(n) in settings and settings[str(n)] == frame['grbl_settings'].get(str(n))
                   == plane['grbl_settings'].get(str(n)) for n in LAYOUT_SETTINGS):
            errors.append('Настройки в калибровке, раскладке и плоскости расходятся')
    layout = frame['frame']['markers_rel_home_machine_mm']
    other = plane['markers_rel_home_machine_mm']
    if (set(layout) != set('0123') or layout.keys()!=other.keys() or frame['frame']['home_id'] != plane['home_id']
            or not all(np.linalg.norm(np.subtract(p,other[k])) <= .5 for k,p in layout.items())):
        errors.append('Раскладка четырёх углов расходится с плоскостью')
    print(f'Свободно {shutil.disk_usage(ROOT).free/2**30:.1f} GiB. Порты: {ports}')
    if args.port.casefold() not in {p.casefold() for p in ports}:
        errors.append(f'Порт {args.port} отсутствует')
    if names is not None:
        print(f'DirectShow: {list(enumerate(names))}')
        if sum(n.casefold() == args.camera_name.casefold() for n in names) != 1:
            errors.append(f'Камера {args.camera_name!r} отсутствует или имя неоднозначно')
    for error in errors:
        print(f'СТОП: {error}')
    return not errors


def apply_calibration_focus(cap, focus):
    try:
        return verify_focus(cap, focus)
    except ValueError as exc:
        raise Abort(str(exc)) from exc


def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_survey_args(ap); add_height_args(ap)
    ap.add_argument('--focus-z-span', type=float, help='Диапазон подъёма от текущей позиции, мм')
    ap.add_argument('--focus-z-step', type=float, default=.25)
    ap.add_argument('--z-feed', type=float, default=30)
    ap.add_argument('--z-up-sign', choices=(-1,1), type=int)
    ap.add_argument('--z-limits', nargs=2, type=float, metavar=('LOW','HIGH'),
                    help='Проверенные границы Z в координатах статуса GRBL, мм')
    ap.add_argument('--tile-step',type=float,default=20)
    ap.add_argument('--ppm',type=float,default=10)
    ap.add_argument('--trial',action='store_true',help='локальная аппаратная проба: квадрат 6 мм, возврат, только c/k/p/v/q')
    ap.add_argument('--preflight',action='store_true',help='проверить файлы и перечень устройств, без открытия камеры и порта')
    ap.add_argument('--camera-name',help='точное имя DirectShow; при пробе в Windows по умолчанию UC60 Video')
    ap.add_argument('--view',type=str,help='показать сохранённый скан, без камеры и станка')
    ap.add_argument('--open',type=str,
                    help='открыть сохранённый скан "как после съёмки": клик/g ведут станок после c -> k -> h')
    args=ap.parse_args()
    if args.view:return view(args.view)
    if args.trial or args.preflight:
        if not args.confirm or args.open or args.no_calib or not (0 < args.probe <= 2):
            ap.error('Проба требует подтверждений, калибровки и хода k до 2 мм')
        if not args.calib:
            try:
                tp = json.loads(args.plane_file.read_text(encoding='utf-8'))
                args.calib = str(ROOT/'calib'/tp['calib'])
            except (OSError, ValueError, KeyError) as exc:
                ap.error(f"Плоскость для пробы: {exc}. Укажите --plane-file и --calib")
        args.width, args.height = args.width or 1920, args.height or 1080
        if not (np.isfinite(args.feed) and np.isfinite(args.slow_feed) and min(args.feed,args.slow_feed)>0):
            ap.error('Подача должна быть положительной')
        args.feed, args.slow_feed = min(args.feed,120),min(args.slow_feed,120)
        if sys.platform == 'win32':
            args.camera_name = args.camera_name or 'UC60 Video'
    names = None
    if args.camera_name:
        if sys.platform != 'win32':
            ap.error('--camera-name использует DirectShow в Windows')
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
    if args.preflight:
        from serial.tools.list_ports import comports
        ok = preflight(args,names,[p.device for p in comports()])
        if not ok:
            raise SystemExit(1)
        print('Файлы и устройства найдены. Фокус, изображение и ответы GRBL проверяются при запуске пробы.')
        return
    if names is not None:
        ids = [i for i,name in enumerate(names) if name.casefold()==args.camera_name.casefold()]
        if len(ids)!=1:
            ap.error(f'Нужна одна камера {args.camera_name!r}. DirectShow видит: {names}')
        args.camera = ids[0]
        print(f'Камера [{args.camera}]: {names[args.camera]}')
    if args.tile_step<=0 or args.ppm<=0:ap.error('step/ppm must be positive')
    if args.head_z_mm is not None and not np.isfinite(args.head_z_mm):ap.error('Некорректная высота головки')
    if args.scan_frames<1 or not np.isfinite(args.scan_settle) or args.scan_settle<0:ap.error('Invalid capture settings')
    cap=open_camera(args.camera,args.width,args.height,fourcc=args.fourcc or None,saved_focus=False)
    tool=None
    try:
        frame=read_frame(cap)
        calib,path=find_calibration(args.camera,frame.shape[1::-1],args.calib)
        if calib is None:raise Abort('Camera calibration required')
        focus = apply_calibration_focus(cap, calib['mode'].get('focus'))
        args.camera_mode = capture_mode(cap, frame.shape[1::-1], args.fourcc)
        args.verified_focus = focus
        title = 'Диагностика у метки [trial]' if args.trial else 'Скан плоскости и высот'
        tool=PlaneScanner(cap,args,WindowUI(title),calib,path)
        tool.focus = focus
        if args.open:
            try:
                tool.open_mosaic(args.open)
            except Abort as e:
                raise SystemExit(f'не открыть мозаику: {e}')
        actions={ord('c'):tool.connect,ord('k'):tool.calibrate_axes,ord('m'):tool.center_and_record,
                 ord('r'):tool.run_survey,ord('t'):tool.measure_table_plane,ord('p'):tool.scan_plane,
                 ord('v'):tool.show_mosaic,ord('h'):tool.go_home,ord('z'):tool.set_zero,
                 ord('j'):tool.scan_focus_z,ord('b'):tool.scan_heights,ord('g'):tool.go_selected}
        if args.trial:
            actions = {k:fn for k,fn in actions.items() if k in map(ord,'ckpv')}
            print('Диагностика у метки: c подключение, k проверка осей, p квадрат 6 мм и возврат, v просмотр, q выход.')
            print('Для обхода r, WASD и полного скана p закройте пробу и запустите скрипт без --trial.')
        else:
            print(__doc__)
        while True:
            frame=read_frame(cap)
            found=detect(tool.detector,cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
            tool.seen_marker |= bool(found)
            key=tool.ui.show(tool.render(frame,found))
            if key==ord('q') or tool.ui.closed:break
            try:
                tool.poll_height_map()
                if key in actions:actions[key]()
                elif key in JOG_STEPS and not args.trial:tool.jog_step=JOG_STEPS[key]
                elif key in JOG_KEYS and not args.trial:tool.jog_key(key)
            except (Abort,GrblError) as e:
                if tool.grbl:tool.grbl.cancel()
                tool.say(str(e))
    finally:
        if tool and tool.grbl:
            try:tool.grbl.cancel()
            finally:tool.grbl.close()
        cap.release();cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Abort as exc:
        raise SystemExit(f'СТОП: {exc}') from None
