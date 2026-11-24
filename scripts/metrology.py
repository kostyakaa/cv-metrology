"""Saved setups and command construction for the metrology terminal."""
import json
import re
import shutil
import math
import sys
from hashlib import sha256
from pathlib import Path
from time import strftime


ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / 'calib'
SETUPS = CALIB / 'setups'
DEFAULT_OPTICS = CALIB / 'calib_cam0_1920x1080_pinhole.json'
DEFAULTS = dict(camera_name='UC60 Video', port='COM4', head_z_mm=None,
                area_mm=[68, 47], area_margin=7, below=18, spans_mm=[283, 93])


def read_config(setup):
    return DEFAULTS | json.loads((setup / 'settings.json').read_text(encoding='utf-8'))


def valid_name(name):
    return re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,39}', name) is not None


def active_setup():
    setup = SETUPS / 'uc60'
    if not (setup / 'settings.json').exists():
        save_setup(setup, DEFAULT_OPTICS, DEFAULTS, seed_geometry=True)
    active = SETUPS / 'active.txt'
    if active.is_file():
        name = active.read_text(encoding='utf-8').strip()
        if valid_name(name) and (SETUPS / name / 'settings.json').is_file():
            return SETUPS / name
    return setup


def camera_args(config):
    if sys.platform == 'win32':
        return ['--camera-name', config['camera_name']]
    index = str(config['camera_name'])
    if not index.isascii() or not index.isdigit():
        raise ValueError('Для этого видеодрайвера укажите числовой индекс камеры в настройках установки.')
    return ['--camera', index]


def number(value, label, low=None, high=None):
    try:
        result = float(value.strip().replace(',', '.'))
    except ValueError:
        raise ValueError(f'{label} — требуется число') from None
    if not math.isfinite(result) or (low is not None and result < low) or (high is not None and result > high):
        raise ValueError(f'{label} — значение вне диапазона')
    return result


def save_setup(setup, optics, config, seed_geometry=False):
    data = optics.read_bytes()
    profile = json.loads(data)
    mode = profile['mode']
    if (profile.get('reliable') is False or mode.get('focus') is None
            or not (profile.get('machine_check') or {}).get('motion')):
        raise ValueError('Нужен достоверный профиль камеры с ручным фокусом и ходом станка')
    snapshot = setup / 'optics' / sha256(data).hexdigest()[:12] / optics.name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        snapshot.write_bytes(data)
    if seed_geometry:
        for name in ('table_frame.json', 'table_plane.json'):
            if (CALIB / name).is_file() and not (setup / name).exists():
                shutil.copy2(CALIB / name, setup / name)
    config = dict(config, optics=str(snapshot.relative_to(setup)))
    target = setup / 'settings.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(target)


def scanner_args(setup, config, profile):
    width, height = profile['image_size']
    args = ['scripts/scan_plane.py', *camera_args(config), '--port', config['port'],
            '--width', str(width), '--height', str(height), '--calib', str(setup / config['optics']),
            '--frame-file', str(setup / 'table_frame.json'), '--plane-file', str(setup / 'table_plane.json'),
            '--area-mm', *map(str, config['area_mm']), '--area-margin', str(config['area_margin']),
            '--below', str(config['below'])]
    span_x, span_y = config['spans_mm']
    args += ['--span-x', str(span_x), '--span-y', str(span_y)]
    if profile['mode'].get('fourcc'):
        args += ['--fourcc', profile['mode']['fourcc']]
    if config['head_z_mm'] is not None:
        args += ['--head-z-mm', str(config['head_z_mm'])]
    return args


def result_paths(action):
    if action in ('mosaic', 'navigate'):
        return sorted((p.parent for p in ROOT.glob('mosaics/**/mosaic.json')), reverse=True)
    if action == 'height':
        return sorted((*ROOT.glob('scans/*/area/area.npz'), *ROOT.glob('bench/*/*/area.npz')), reverse=True)
    return sorted((p.parent for p in ROOT.glob('scans/*/scan.json')), reverse=True)


def command(action, setup, path=None, square='', focus=''):
    config = read_config(setup)
    python = [sys.executable, '-u']
    if action == 'mosaic':
        return python + ['scripts/scan_plane.py', '--view', str(path)]
    if action == 'height':
        return python + ['z/view_height_map.py', str(path)]
    if action == 'analyze':
        shell = shutil.which('powershell.exe' if sys.platform == 'win32' else 'pwsh')
        if not shell:
            raise ValueError('Для офлайн-пересчёта установите PowerShell (pwsh).')
        meta = json.loads((path / 'scan.json').read_text(encoding='utf-8'))
        plan = meta.get('scan_plan', {})
        x, y = plan.get('area_mm') or config['area_mm']
        margin = plan.get('area_margin', config['area_margin'])
        below = meta.get('height_settings', {}).get('below', config['below'])
        out = ROOT / 'bench' / f'{path.name}-{strftime("%Y%m%d-%H%M%S")}'
        return [shell, '-NoProfile', '-File', str(ROOT / 'scripts/run_metrology.ps1'),
                '-Stage', 'Analyze', '-Python', sys.executable, '-RunDir', str(out), '-Scan', str(path),
                '-AreaX', str(x), '-AreaY', str(y), '-Margin', str(margin), '-Below', str(below)]
    profile = json.loads((setup / config['optics']).read_text(encoding='utf-8'))
    if action == 'calibrate':
        camera = camera_args(config)
        if camera[0] == '--camera-name':
            from pygrabber.dshow_graph import FilterGraph
            devices = FilterGraph().get_input_devices()
            matches = [i for i, name in enumerate(devices) if name.casefold() == camera[1].casefold()]
            if len(matches) != 1:
                raise ValueError(f'Уточните имя камеры. DirectShow обнаружил {devices}')
            camera = ['--camera', str(matches[0])]
        w, h = profile['image_size']
        mode = profile['mode']
        pitch = ['--square-mm', str(number(square, 'Шаг клетки', 0.000001))] if square.strip() else []
        focus_args = []
        if focus.strip():
            value = number(focus, 'Фокус', 0, 1023)
            if not value.is_integer():
                raise ValueError('Фокус задаётся целым числом')
            focus_args = ['--focus', str(int(value))]
        return python + ['scripts/calibrate_distortion.py', *camera, '--width', str(w), '--height', str(h),
                         '--fourcc', mode.get('fourcc') or '', *focus_args,
                         '--grbl', config['port'], '--feed', '120', *pitch]
    flags = {'session': [], 'auto': ['--auto'], 'check': ['--preflight'], 'navigate': ['--open', str(path)]}
    return python + scanner_args(setup, config, profile) + flags[action]


if __name__ == '__main__':
    from metrology_tui import MetrologyApp
    MetrologyApp(active_setup()).run()


