# cv-metrology
CV интерфейс для измерительной микроскопии

## Установка
```
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
```
`pygrabber` и `comtypes` ставятся только на Windows (DirectShow); на Linux не нужны.

## Linux
Работает через V4L2 (камера) и pyserial (станок). Windows - через DirectShow, как раньше.

- **Пакеты системы:** `v4l-utils` (нужен `scripts/list_formats.py`, `scripts/compare_modes.py`, `scripts/camera_params.py`),
  Tk для окна 3D (`tkinter`), любой шрифт с кириллицей (DejaVu/Liberation; иначе берётся DejaVu из matplotlib).
- **Права:** камера - группа `video`, станок - группа `uucp` (Arch) или `dialout` (Debian/Ubuntu); после добавления
  в группу нужно перелогиниться. Проверка: `id`.
- **Камеры:** `.venv/bin/python scripts/list_cameras.py` - список камер Linux (`/dev/videoN`) и окно с каждой.
  Узлы метаданных пропускаются: у одной камеры их обычно два (видео и метаданные).
- **Номер камеры проекта:** калибровки и фокус привязаны к номеру (`calib/calib_cam0_*`, `calib/camera_params_0.json`),
  а `/dev/videoN` зависит от порядка подключения. `calib/camera_map.json` связывает номер проекта с частью имени
  устройства: `{"0": "UC60"}` - `--camera 0` всегда открывает UC60. Номера вне файла - это `/dev/video<N>` как есть.
- **Формат:** без `--fourcc` камера отдаёт сырой YUYV (как YUY2 на Windows); на 1080p у UC60 это ~5 кадров/с.
  Сжатый MJPG (30 кадров/с) - `--fourcc MJPG`, но калибровка верна только для режима, в котором снята.
- **Порт станка:** по умолчанию первый `/dev/ttyUSB*` или `/dev/ttyACM*` (на Windows - `COM4`), иначе `--port /dev/ttyUSB0`.
  В `calibrate_distortion.py` флаг `--grbl` без значения берёт порт по умолчанию.

## Запуск
```
.venv/bin/python scripts/calibrate_distortion.py --camera 0 --board 10mm --width 1920 --height 1080 --grbl
.venv/bin/python z/measure_height.py --camera 0 --width 1920 --height 1080
.venv/bin/python z/measure_height.py --from-scan scans/scan_<время>      # разбор скана, без станка и камеры
.venv/bin/python z/plot_3d.py scans/scan_<время> --view model --show     # 3D-окно, вращать мышью
.venv/bin/python z/make_model.py scans/scan_<время>                      # 2.5D-модель, STL
```
