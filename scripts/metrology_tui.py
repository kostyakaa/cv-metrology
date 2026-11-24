"""Keyboard and mouse console for acquisition and offline metrology."""
import asyncio
import json
import os
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, OptionList, RichLog, Static
from textual.widgets.option_list import Option

import metrology as project


# ID, palette label, context shown before starting the command.
ACTIONS = [
    ('session', 'Открыть рабочую сессию', 'Камера и сохранённая геометрия. На видимой метке нажмите c → k. '
     'Клавиша p снимает поле, b строит карту высот вокруг выбранного XY.'),
    ('auto', 'Полный цикл по четырём меткам', 'k → r → t → p. В окне камеры нажмите c и подтвердите маршрут.'),
    ('check', 'Проверить установку', 'Проверяет файлы калибровки и перечень устройств без открытия порта.'),
    ('mosaic', 'Посмотреть мозаику', 'Открывает сохранённый снимок поля для просмотра.'),
    ('navigate', 'Наведение по мозаике', 'Открывает сохранённую мозаику в рабочей сессии с проверкой привязки.'),
    ('height', 'Посмотреть карту высот', 'Открывает измеренную карту из съёмки или офлайн-пересчёта.'),
    ('analyze', 'Пересчитать стереокарту', 'Использует снимки и размеры области из выбранного скана. Результат сохраняется в bench.'),
    ('calibrate', 'Калибровка по ChArUco', 'Съёмка доски для выбранной камеры. Шаг клетки можно задать по измеренной длине.'),
    ('setups', 'Выбрать установку', 'Переключает сохранённые настройки, раскладку и плоскость.'),
    ('edit', 'Настроить установку', 'Изменяет параметры текущей установки.'),
    ('new', 'Создать установку', 'Копирует параметры и оптику. Геометрия новой оснастки измеряется отдельно.'),
]


class Fields(ModalScreen):
    BINDINGS = [('escape', 'cancel', 'Назад')]

    def __init__(self, title, fields, submit):
        super().__init__()
        self.caption, self.fields, self.submit = title, fields, submit

    def compose(self) -> ComposeResult:
        with Vertical(id='dialog'):
            yield Label(self.caption, classes='heading')
            with VerticalScroll(id='fields'):
                for key, label, value in self.fields:
                    yield Label(label)
                    yield Input(value=str(value), id=key)
            yield Static('', id='error', markup=False)
            with Horizontal(id='buttons'):
                yield Button('Назад', id='cancel')
                yield Button('Применить', id='apply', variant='primary')

    def on_mount(self):
        self.query_one(Input).focus()

    def action_cancel(self):
        self.dismiss(None)

    @on(Button.Pressed, '#cancel')
    def cancel(self):
        self.action_cancel()

    @on(Input.Submitted)
    def next_field(self, event):
        inputs = list(self.query(Input))
        index = inputs.index(event.input)
        if index == len(inputs) - 1:
            self.apply()
        else:
            inputs[index + 1].focus()

    @on(Button.Pressed, '#apply')
    def apply(self):
        values = {field.id: field.value.strip() for field in self.query(Input)}
        try:
            result = self.submit(values)
        except (ValueError, OSError, KeyError, ImportError) as exc:
            self.query_one('#error', Static).update(str(exc))
        else:
            self.dismiss(result)


class MetrologyApp(App):
    ENABLE_COMMAND_PALETTE = False
    TITLE = 'CV Metrology'
    BINDINGS = [
        Binding('ctrl+k', 'search', 'Поиск'),
        Binding('escape', 'back', 'Назад'),
        Binding('f3', 'journal', 'Журнал'),
        Binding('ctrl+q', 'quit', 'Выход', priority=True),
    ]
    CSS = """
    Screen { background: #11171a; color: #e0e8e8; }
    #bar { height: 3; padding: 1 2 0 2; color: #7edcc5; text-style: bold; }
    #workspace { height: 1fr; padding: 0 1; }
    #choices { width: 3fr; padding: 0 1; }
    #context { width: 2fr; border-left: solid #34454a; padding: 0 2; }
    .heading { color: #7edcc5; text-style: bold; margin: 1 0 0 0; }
    #detail, #profile { height: auto; padding-top: 1; }
    #search { height: 3; margin: 0; }
    OptionList { height: 1fr; border: none; background: #11171a; }
    OptionList > .option-list--option-highlighted { background: #204b48; color: white; text-style: bold; }
    #status { height: auto; max-height: 3; padding: 0 2; color: #a9c8bf; }
    RichLog { margin: 0 2; height: 1fr; background: #11171a; }
    Footer { background: #1c282b; }
    Fields { align: center middle; background: #0008; }
    #dialog { background: #1c282b; border: round #73bea9; padding: 0 2; width: 72; max-width: 96%;
              height: auto; max-height: 95%; }
    #fields { height: auto; max-height: 65vh; }
    #buttons { height: 3; align-horizontal: right; }
    #error { height: auto; color: #ffba9c; }
    Screen.compact #context { display: none; }
    """

    def __init__(self, setup):
        super().__init__()
        self.setup = setup
        self.page = 'home'
        self.entries = []
        self.visible_entries = []
        self.in_journal = False
        self.running = None

    def compose(self) -> ComposeResult:
        yield Static(id='bar', markup=False)
        with Horizontal(id='workspace'):
            with Vertical(id='choices'):
                yield Input(placeholder='Найти команду…', id='search')
                yield OptionList(id='list', markup=False)
            with VerticalScroll(id='context'):
                yield Label('Установка', classes='heading')
                yield Static(id='profile', markup=False)
                yield Label('Действие', classes='heading')
                yield Static(id='detail', markup=False)
        yield RichLog(id='log', wrap=True, markup=False, max_lines=4000)
        yield Static(id='status', markup=False)
        yield Footer()

    def on_mount(self):
        self.query_one('#log').display = False
        self.home()

    def on_resize(self, event):
        self.screen.set_class(event.size.width < 72, 'compact')

    def home(self):
        config = project.read_config(self.setup)
        try:
            profile = json.loads((self.setup / config['optics']).read_text(encoding='utf-8'))
            w, h = profile['image_size']
            mode = f"{w} × {h}   Фокус {profile['mode']['focus']}"
        except (KeyError, OSError, ValueError) as exc:
            mode = f'Проверьте профиль оптики в настройках: {exc}'
        self.query_one('#profile', Static).update(
            f"{config['camera_name']}   {config['port']}\n{mode}\nОбласть {config['area_mm'][0]:g} × {config['area_mm'][1]:g} мм")
        self.set_page('home', 'Команды', ACTIONS)

    def set_page(self, page, title, entries):
        self.page, self.entries = page, entries
        self.query_one('#bar', Static).update(f'CV Metrology   /   {self.setup.name}   /   {title}')
        search = self.query_one('#search', Input)
        search.placeholder = 'Найти команду…' if page == 'home' else 'Фильтр по названию…'
        search.value = ''
        self.filter('')
        self.query_one('#list').focus()

    def filter(self, value):
        words = value.casefold().split()
        self.visible_entries = [e for e in self.entries if all(w in e[1].casefold() for w in words)]
        listing = self.query_one('#list', OptionList)
        listing.clear_options()
        listing.add_options([Option(Text(entry[1])) for entry in self.visible_entries])
        listing.highlighted = 0 if self.visible_entries else None
        self.describe()
        if not self.visible_entries:
            self.status('Совпадений нет.' if self.entries else 'Сохранённых результатов пока нет. Esc — назад.')
        else:
            self.status('↑ ↓ выбрать   Enter открыть   / поиск')

    @on(Input.Changed, '#search')
    def search_changed(self, event):
        self.filter(event.value)

    def on_key(self, event):
        if isinstance(self.screen, Fields) or self.running or self.in_journal:
            return
        search = self.query_one('#search', Input)
        if self.focused is search and event.key in ('up', 'down'):
            listing = self.query_one('#list', OptionList)
            listing.focus()
            if event.key == 'down':
                listing.action_cursor_down()
            else:
                listing.action_cursor_up()
            event.stop()
        elif isinstance(self.focused, OptionList) and event.character and event.character.isprintable():
            search.focus()
            if event.character != '/':
                search.insert_text_at_cursor(event.character)
            event.stop()

    @on(OptionList.OptionHighlighted)
    def describe(self):
        index = self.query_one('#list', OptionList).highlighted
        desc = self.visible_entries[index][2] if index is not None and index < len(self.visible_entries) else ''
        self.query_one('#detail', Static).update(desc)

    @on(OptionList.OptionSelected)
    @on(Input.Submitted, '#search')
    def select_entry(self):
        index = self.query_one('#list', OptionList).highlighted
        if index is None or self.running or not self.visible_entries:
            return
        item, label, _ = self.visible_entries[index]
        try:
            if self.page == 'setups':
                self.switch_setup(item)
            elif self.page != 'home':
                self.launch(self.page, item)
            elif item in ('mosaic', 'height', 'navigate', 'analyze'):
                self.set_page(item, label, [(p, str(p.relative_to(project.ROOT)), label) for p in project.result_paths(item)])
            elif item == 'setups':
                saved = sorted(p.parent for p in project.SETUPS.glob('*/settings.json'))
                self.set_page(item, label, [(p, p.name, str(p)) for p in saved])
            elif item in ('edit', 'new'):
                self.edit_setup(item == 'new')
            elif item == 'calibrate':
                self.push_screen(Fields(label, [('focus', 'Фокус съёмки (пусто — из настроек камеры)', ''),
                                               ('square', 'Шаг клетки, мм (пусто — номинал)', '')],
                                        lambda v: project.command(item, self.setup, square=v['square'], focus=v['focus'])),
                                 lambda args: self.start_job(label, args) if args else None)
            else:
                self.launch(item)
        except (OSError, ValueError, KeyError, ImportError) as exc:
            self.status(str(exc))

    def switch_setup(self, setup):
        if setup is None:
            return
        (project.SETUPS / 'active.txt').write_text(setup.name, encoding='utf-8')
        self.setup = setup
        self.home()

    def edit_setup(self, new):
        cfg = project.read_config(self.setup)
        optics = self.setup / cfg['optics']
        fields = [('name', 'Имя новой установки', '')] if new else []
        fields += [('camera', 'Имя камеры в Windows или индекс в Linux', cfg['camera_name']),
                   ('port', 'Порт GRBL', cfg['port']), ('optics', 'Файл калибровки', optics),
                   ('head', 'Положение головки Z, мм (пусто — не задано)', '' if cfg['head_z_mm'] is None else cfg['head_z_mm']),
                   ('area', 'Область скана X Y, мм', ' '.join(map(str, cfg['area_mm']))),
                   ('spans', 'Между центрами меток X Y, мм', ' '.join(map(str, cfg['spans_mm']))),
                   ('margin', 'Запас вокруг детали, мм', cfg['area_margin']),
                   ('below', 'Глубина под плоскостью, мм', cfg['below'])]

        def save(values):
            target = self.setup
            if new:
                name = values['name']
                if not project.valid_name(name):
                    raise ValueError('Имя установки — до 40 латинских букв, цифр, знаков - и _.')
                target = project.SETUPS / name
                if (target / 'settings.json').exists():
                    raise ValueError('Установка с таким именем уже существует.')
            if not values['port'] or not values['camera']:
                raise ValueError('Укажите камеру и порт.')
            config = dict(cfg, camera_name=values['camera'], port=values['port'])
            for field, key in [('area', 'area_mm'), ('spans', 'spans_mm')]:
                parts = values[field].split()
                if len(parts) != 2:
                    raise ValueError('Введите X и Y через пробел.')
                config[key] = [project.number(p, 'Размер XY', .000001, 400) for p in parts]
            config.update(head_z_mm=project.number(values['head'], 'Головка Z') if values['head'] else None,
                          area_margin=project.number(values['margin'], 'Запас', 0, 50),
                          below=project.number(values['below'], 'Глубина', 0, 100))
            source = Path(values['optics'].strip('"')).expanduser()
            project.save_setup(target, project.ROOT / source, config)
            return target

        self.push_screen(Fields('Новая установка' if new else self.setup.name, fields, save), self.switch_setup)

    def status(self, text):
        self.query_one('#status', Static).update(text)

    def launch(self, action, path=None):
        label = next(label for key, label, _ in ACTIONS if key == action)
        self.start_job(label, project.command(action, self.setup, path))

    def start_job(self, label, args):
        if self.running:
            return
        self.running = label
        self.set_journal(True)
        self.query_one(RichLog).write(Text(f'› {label}', style='bold cyan'))
        self.status(f'{label} выполняется. Клавиши управления съёмкой работают в окне камеры.')
        self.execute(label, args)

    @work
    async def execute(self, label, args):
        log = self.query_one(RichLog)
        try:
            process = await asyncio.create_subprocess_exec(
                *args, cwd=project.ROOT, env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1'),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            async for line in process.stdout:
                log.write(Text(line.decode('utf-8', errors='replace').rstrip()))
            code = await process.wait()
            text = f'{label} — завершено' if code == 0 else f'{label} — код выхода {code}'
        except OSError as exc:
            text = f'{label} — {exc}'
        finally:
            self.running = None
        log.write(Text(text, style='bold'))
        self.status(text + '. Esc — к командам.')

    def set_journal(self, visible):
        self.in_journal = visible
        self.query_one('#workspace').display = not visible
        self.query_one('#log').display = visible
        if visible:
            self.query_one('#bar', Static).update(f'CV Metrology   /   {self.setup.name}   /   Журнал')
            self.query_one('#log').focus()
        else:
            self.home()

    def action_journal(self):
        if not self.running and not isinstance(self.screen, Fields):
            self.set_journal(not self.in_journal)

    def action_search(self):
        if not self.running and not isinstance(self.screen, Fields):
            if self.in_journal:
                self.set_journal(False)
            self.query_one('#search').focus()

    def action_back(self):
        if self.running:
            return
        if self.in_journal:
            self.set_journal(False)
        elif self.query_one('#search', Input).value:
            self.query_one('#search', Input).value = ''
        else:
            self.home()

    def action_quit(self):
        if self.running:
            self.status(f'Сначала завершите «{self.running}». Для съёмки нажмите q в окне камеры.')
        else:
            self.exit()
