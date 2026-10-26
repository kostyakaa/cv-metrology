"""Минимальный клиент GRBL 1.1 для XY-стола: статус, jog-перемещения, отмена.

Все перемещения - jog-команды ($J=...): GRBL их принимает в буфер и отвечает ok,
статус во время хода - Jog, по окончании - Idle. Jog можно мгновенно отменить
(realtime-байт 0x85: плавная остановка с сохранением позиции) и он не меняет
модальное состояние G-кода (G90/G91, подача) - безопасно для ручной отладки.
"""
import time

JOG_CANCEL = b"\x85"
SOFT_RESET = b"\x18"


class GrblError(RuntimeError):
    pass


class GrblAborted(RuntimeError):
    pass


class Grbl:
    def __init__(self, port=None, baud=115200, link=None, reset_wait=2.0, log=print):
        """link - готовый объект с write()/readline() (для тестов); иначе открывается serial-порт.
        Открытие порта на Arduino перезагружает GRBL - станок при этом не двигается."""
        if link is None:
            import serial
            link = serial.Serial(port, baud, timeout=0.1)
        self.link = link
        self.log = log
        self.state = None
        self.mpos = self.wpos = self.wco = None
        self.banner = self._wait_banner(reset_wait)
        if not self.banner.startswith("Grbl 1.1"):
            raise GrblError(f"Нужен GRBL 1.1 (jog-команды $J=), а ответил: {self.banner!r}")
        self.status()

    # --- обмен
    def _readline(self):
        raw = self.link.readline()
        return raw.decode(errors="replace").strip() if raw else None

    def _handle_async(self, line):
        if line.startswith("<"):
            self._parse_status(line)
        elif line.startswith("ALARM"):
            self.state = "Alarm"
            self.log(f"GRBL: {line}")
        elif line:
            self.log(f"GRBL: {line}")

    def _wait_banner(self, reset_wait):
        deadline = time.time() + reset_wait + 1.0
        while time.time() < deadline:
            line = self._readline()
            if line and line.startswith("Grbl"):
                return line
        self.link.write(SOFT_RESET)  # плата без авто-сброса при открытии порта: просим баннер сбросом (станок стоит)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            line = self._readline()
            if line and line.startswith("Grbl"):
                return line
        raise GrblError("GRBL не ответил баннером - проверьте порт и скорость")

    def cmd(self, line, timeout=5.0):
        self.link.write((line + "\n").encode())
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self._readline()
            if resp is None:
                continue
            if resp == "ok":
                return
            if resp.startswith("error"):
                raise GrblError(f"{line!r} -> {resp}")
            self._handle_async(resp)
            if self.state == "Alarm":
                raise GrblError(f"{line!r} -> ALARM")
        raise GrblError(f"нет ответа на {line!r}")

    def _parse_status(self, line):
        fields = line.strip("<>").split("|")
        self.state = fields[0].split(":")[0]
        for f in fields[1:]:
            key, _, val = f.partition(":")
            if key in ("MPos", "WPos", "WCO"):
                xyz = [float(v) for v in val.split(",")]
                setattr(self, key.lower(), xyz)

    def status(self, timeout=1.0):
        self.link.write(b"?")
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._readline()
            if line is None:
                continue
            if line.startswith("<"):
                self._parse_status(line)
                return self.state
            self._handle_async(line)
        raise GrblError("нет ответа на '?'")

    def position(self):
        """XY для относительных расчётов: MPos (по умолчанию, $10=1) или WPos ($10=0) - всегда одна и та же
        система (WCO в отчёте приходит не каждый раз, смешивать нельзя)."""
        self.status()
        p = self.mpos if self.mpos is not None else self.wpos
        if p is None:
            raise GrblError("GRBL не сообщает позицию")
        return p[0], p[1]

    # --- движение
    def require_idle(self):
        state = self.status()
        if state != "Idle":
            raise GrblError(f"станок не в Idle, а в {state} (Alarm - разблокировка $X, Hold - продолжение ~)")

    def jog(self, dx, dy, feed, tick=None):
        """Относительный ход XY и ожидание конца. tick() вызывается во время хода; True -> отмена."""
        self.require_idle()
        self.cmd(f"$J=G21G91X{dx:.3f}Y{dy:.3f}F{feed:.0f}")
        self.wait_idle(tick, timeout=60.0 * max(abs(dx), abs(dy)) / feed * 1.5 + 5.0)

    def wait_idle(self, tick=None, timeout=60.0):
        deadline = time.time() + timeout
        while True:
            if tick is not None and tick():
                self.cancel()
                raise GrblAborted("ход отменён")
            state = self.status()
            if state == "Idle":
                return
            if state == "Alarm":
                raise GrblError("ALARM во время хода - позиция могла сбиться")
            if time.time() > deadline:
                self.cancel()
                raise GrblError("ход не закончился вовремя - отменён")
            time.sleep(0.05)

    def cancel(self):
        """Плавная остановка jog с сохранением позиции."""
        self.link.write(JOG_CANCEL)
        for _ in range(100):
            if self.status() in ("Idle", "Alarm"):
                return
            time.sleep(0.05)

    def settings(self, timeout=5.0):
        """Настройки GRBL ($$) -> {номер: значение}."""
        self.link.write(b"$$\n")
        out = {}
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._readline()
            if line is None:
                continue
            if line == "ok":
                return out
            if line.startswith("error"):
                raise GrblError(f"'$$' -> {line}")
            if line.startswith("$") and "=" in line:
                key, _, val = line[1:].partition("=")
                out[int(key)] = float(val.split()[0])
            else:
                self._handle_async(line)
        raise GrblError("нет ответа на '$$'")

    def unlock(self):
        self.cmd("$X")

    def set_work_zero_here(self):
        """Текущая точка = ноль рабочей СК G54 (хранится в EEPROM GRBL)."""
        self.cmd("G10 L20 P1 X0 Y0")

    def close(self):
        if hasattr(self.link, "close"):
            self.link.close()
