"""Мозаика плоскости стола: съёмка поля змейкой (p), выбор точки кликом, переезд станка (g).

Съёмка: c подключение, k оси + привязка к раскладке меток, t плоскость стола, p скан,
v окно мозаики. Координаты - мм команды станка относительно домашней метки, не
сертифицированные физические мм.

Сохранённый скан открывается "как после съёмки" - без повторного сканирования:
    python scripts/scan_plane.py --camera 0 --width 1920 --height 1080 --port COM4 --open mosaics/<сеанс>
Порядок в открытой сессии (домашняя точка по метке, концевиков нет):
    камеру на любую метку -> c -> k (оси + привязка) -> h (проверка: к домашней метке)
    -> v (мозаика) -> клик -> g.
g едет только внутри зоны, где станок уже был. Если с момента съёмки мозаики менялись
$3/$100/$101 - скрипт это поймает при подключении и мозаику отключит.
--view mosaics/<сеанс> - только посмотреть, без камеры и станка.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'z'), str(ROOT/'scripts')]
from common.camera import open_camera, read_frame
from common.grbl import GrblError
from common.markers import detect
from common.mosaic import Mosaic, pixel_to_machine
from measure_height import HeightTool, add_height_args
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
        self.say('v - окно мозаики (клик/g). g ведёт станок после привязки: камеру на любую метку -> c -> k -> h')

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
        super().connect()
        try:
            cur = self.grbl.settings()
        except GrblError:
            return
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
            raise Abort('First k on a visible marker')
        exp = self.expected()
        if len(exp) != 4:
            raise Abort('Four surveyed markers required (r)')
        try:
            settings = {str(n): v for n, v in self.grbl.settings().items() if n in LAYOUT_SETTINGS}
        except GrblError:
            settings = {}
        start = self.position()
        plane, why = self.table_plane_here(start)
        if plane is None:
            raise Abort(why + '; use t first')
        polygon = cv2.convexHull(np.array(list(exp.values()),np.float32)).reshape(-1,2)
        home = np.array(exp[self.home_id()])
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
        if len(targets)>1000:
            raise Abort('Too many tiles')
        # Validate ALL endpoints, including the return, before beginning.
        for p in targets+[start]:
            self.check_target(p)
        if not self.confirm(f'Scan {len(targets)} stops, step <= {step:.1f} command mm, inside marker polygon; clear entire XY path'):
            return
        folder = ROOT/'mosaics'/time.strftime('%Y%m%d-%H%M%S')
        folder.mkdir(parents=True,exist_ok=False)
        mosaic = Mosaic(polygon-home,self.args.ppm)
        meta = dict(version=1, units='command_mm_relative_to_home_marker', home_id=self.home_id(),
                    home_session=home.tolist(), polygon=(polygon-home).tolist(), ppm=self.args.ppm,
                    camera_matrix=self.K.tolist(), dist_coeffs=self.D.tolist(), focus=self.focus,
                    calibration=self.calib_name, model=self.calib_model, blend='central-source',
                    grbl_settings=settings, records=[], complete=False)
        def save():
            meta['pair_errors'] = mosaic.errors
            meta['coverage_fraction'] = float(np.count_nonzero((mosaic.weight>0)&(mosaic.region>0))/max(1,np.count_nonzero(mosaic.region)))
            (folder/'mosaic.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
            cv2.imwrite(str(folder/'mosaic.png'),mosaic.image())
            cv2.imwrite(str(folder/'coverage.png'),(mosaic.weight>0).astype(np.uint8)*255)
        try:
            marker = self.require_marker()
            delta = targets[0]-self.position()
            if np.linalg.norm(delta)>self.args.probe*2:
                self.verify_direction(marker['id'],delta)
            for i,p in enumerate(targets):
                self.move(*(p-self.position()),feed=min(self.args.feed,600))
                raw = self.capture_color()
                pos = self.position()
                plane,why = self.table_plane_here(pos)
                if plane is None:
                    raise Abort(why)
                H = pixel_to_machine(self.K,self.V,*plane,pos-home)
                filename = f'frame_{i:04d}.png'
                cv2.imwrite(str(folder/filename),raw)
                meta['records'].append(dict(file=filename,position=(pos-home).tolist(),H=H.tolist()))
                mosaic.add(self.rect(raw),self.rect.valid,H,pos-home,raw=raw,K=self.K,D=self.D,model=self.calib_model)
                save()
                self.say(f'Tile {i+1}/{len(targets)}, coverage {meta["coverage_fraction"]:.1%}')
            self.move(*(start-self.position()),feed=min(self.args.feed,600))
            _,closure = self.center_on(marker['id'])
            meta['end_marker_position'] = (closure-home).tolist()
            meta['closure_command_mm'] = (closure-np.array(exp[marker['id']])).tolist()
            meta['complete'] = True
        finally:
            # On failure stay stopped; never blindly return after loss of position.
            save()
        self.mosaic_folder = folder
        self.mosaic_meta = meta
        self.mosaic_home_id = int(meta['home_id'])
        self.show_mosaic()

    def show_mosaic(self):
        if self.mosaic_folder is None:
            raise Abort('мозаики нет: p - снять, --open <сеанс> - открыть сохранённую')
        view(self.mosaic_folder, self)


def view(folder, tool=None):
    folder, meta = load_mosaic(folder)
    image=cv2.imread(str(folder/'mosaic.png'))
    valid=cv2.imread(str(folder/'coverage.png'),0)
    origin=np.min(meta['polygon'],axis=0)
    scale=min(1.,1200/image.shape[1],800/image.shape[0])
    viewport=[min(1200,image.shape[1]),min(800,image.shape[0])]
    offset=np.zeros(2,dtype=float)
    drag=[]
    selected=[]
    name='Plane mosaic (click: coordinates, g: move, q: close)'
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
        if not(0<=v<valid.shape[0] and 0<=u<valid.shape[1]) or not valid[v,u]:return
        xy=origin+np.array([u,v])/meta['ppm']
        selected[:] = [xy]
        print(f'Home-relative command XY: {xy[0]:.3f}, {xy[1]:.3f} mm')
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
            if key==ord('g') and tool is not None and selected:
                if anchor is None:
                    print('мозаика не привязана к станку: камеру на любую метку -> c -> k (-> h), снова v и g')
                    continue
                target=selected[0]+anchor
                try:
                    tool.check_target(target)
                    if tool.confirm(f'Camera to mosaic XY {selected[0].round(3).tolist()} '
                                    f'(machine X{target[0]:.2f} Y{target[1]:.2f})?'):
                        bc=tool.backlash_comp
                        tool.move(*(target-tool.position()-bc),feed=min(tool.args.feed,600))
                        tool.move(bc,bc,feed=tool.args.slow_feed,check=False)
                        p=tool.position()
                        print(f'камера на точке: X{p[0]:.3f} Y{p[1]:.3f} (станок); проверьте по живому кадру')
                except (Abort,GrblError) as e:
                    if tool.grbl:tool.grbl.cancel()
                    print(f'СТОП: {e}')
    finally:
        cv2.destroyWindow(name)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    add_survey_args(ap); add_height_args(ap)
    ap.add_argument('--tile-step',type=float,default=20)
    ap.add_argument('--ppm',type=float,default=10)
    ap.add_argument('--view',type=str,help='показать сохранённый скан, без камеры и станка')
    ap.add_argument('--open',type=str,
                    help='открыть сохранённый скан "как после съёмки": клик/g ведут станок после c -> k -> h')
    args=ap.parse_args()
    if args.view:return view(args.view)
    if args.tile_step<=0 or args.ppm<=0:ap.error('step/ppm must be positive')
    if args.scan_frames<1 or args.scan_settle<0:ap.error('Invalid capture settings')
    cap=open_camera(args.camera,args.width,args.height,fourcc=args.fourcc or None)
    tool=None
    try:
        frame=read_frame(cap)
        calib,path=find_calibration(args.camera,frame.shape[1::-1],args.calib)
        if calib is None:raise Abort('Camera calibration required')
        tool=PlaneScanner(cap,args,WindowUI('Plane scan'),calib,path)
        if tool.focus != calib.get('mode',{}).get('focus'):
            raise Abort('Focus differs from calibration')
        if args.open:
            try:
                tool.open_mosaic(args.open)
            except Abort as e:
                raise SystemExit(f'не открыть мозаику: {e}')
        actions={ord('c'):tool.connect,ord('k'):tool.calibrate_axes,ord('m'):tool.center_and_record,
                 ord('r'):tool.run_survey,ord('t'):tool.measure_table_plane,ord('p'):tool.scan_plane,
                 ord('v'):tool.show_mosaic,ord('h'):tool.go_home,ord('z'):tool.set_zero}
        print(__doc__)
        while True:
            frame=read_frame(cap)
            found=detect(tool.detector,cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
            tool.seen_marker |= bool(found)
            key=tool.ui.show(tool.render(frame,found))
            if key==ord('q') or tool.ui.closed:break
            try:
                if key in actions:actions[key]()
                elif key in JOG_STEPS:tool.jog_step=JOG_STEPS[key]
                elif key in JOG_KEYS:tool.jog_key(key)
            except (Abort,GrblError) as e:
                if tool.grbl:tool.grbl.cancel()
                tool.say(str(e))
    finally:
        if tool and tool.grbl:
            try:tool.grbl.cancel()
            finally:tool.grbl.close()
        cap.release();cv2.destroyAllWindows()


if __name__=='__main__':main()
