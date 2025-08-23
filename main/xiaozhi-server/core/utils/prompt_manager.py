"""
Модуль управления системными подсказками (промптами)
Отвечает за загрузку, кэширование и динамическое обновление системных промптов,
включая быструю инициализацию и расширенную сборку контекста
"""

import os
import cnlunar
from typing import Dict, Any
from config.logger import setup_logging
from jinja2 import Template

TAG = __name__

# Сопоставление английских дней недели с русскими
WEEKDAY_MAP = {
    "Monday": "понедельник",
    "Tuesday": "вторник",
    "Wednesday": "среда",
    "Thursday": "четверг",
    "Friday": "пятница",
    "Saturday": "суббота",
    "Sunday": "воскресенье",
}

# Список эмодзи (можно использовать в ответах)
EMOJI_List = [
    "😶", "🙂", "😆", "😂", "😔", "😠", "😭", "😍", "😳", "😲",
    "😱", "🤔", "😉", "😎", "😌", "🤤", "😘", "😏", "😴", "😜", "🙄"
]


class PromptManager:
    """Менеджер системных промптов — управляет и обновляет подсказки для ИИ"""

    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or setup_logging()
        self.base_prompt_template = None  # Шаблон основного промпта
        self.last_update_time = 0

        # Подключаем глобальный менеджер кэша
        from core.utils.cache.manager import cache_manager, CacheType

        self.cache_manager = cache_manager
        self.CacheType = CacheType

        self._load_base_template()

    def _load_base_template(self):
        """Загружает базовый шаблон промпта из файла"""
        try:
            template_path = "agent-base-prompt.txt"
            cache_key = f"prompt_template:{template_path}"

            # Сначала проверяем кэш
            cached_template = self.cache_manager.get(self.CacheType.CONFIG, cache_key)
            if cached_template is not None:
                self.base_prompt_template = cached_template
                self.logger.bind(tag=TAG).debug("Шаблон промпта загружен из кэша")
                return

            # Если в кэше нет — читаем из файла
            if os.path.exists(template_path):
                with open(template_path, "r", encoding="utf-8") as f:
                    template_content = f.read()

                # Сохраняем в кэш (кэш типа CONFIG не истекает автоматически)
                self.cache_manager.set(
                    self.CacheType.CONFIG, cache_key, template_content
                )
                self.base_prompt_template = template_content
                self.logger.bind(tag=TAG).debug("Базовый шаблон промпта успешно загружен и закэширован")
            else:
                self.logger.bind(tag=TAG).warning("Файл agent-base-prompt.txt не найден")
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Ошибка загрузки шаблона промпта: {e}")

    def get_quick_prompt(self, user_prompt: str, device_id: str = None) -> str:
        """Быстрое получение системного промпта (с использованием кэша устройства)"""
        device_cache_key = f"device_prompt:{device_id}"
        cached_device_prompt = self.cache_manager.get(
            self.CacheType.DEVICE_PROMPT, device_cache_key
        )
        if cached_device_prompt is not None:
            self.logger.bind(tag=TAG).debug(f"Используется кэшированный промпт для устройства {device_id}")
            return cached_device_prompt
        else:
            self.logger.bind(tag=TAG).debug(
                f"Устройство {device_id} не имеет кэшированного промпта, используется переданный текст"
            )

        # Кэшируем переданный промпт (если указан device_id)
        if device_id:
            device_cache_key = f"device_prompt:{device_id}"
            self.cache_manager.set(self.CacheType.CONFIG, device_cache_key, user_prompt)
            self.logger.bind(tag=TAG).debug(f"Промпт для устройства {device_id} сохранён в кэш")

        self.logger.bind(tag=TAG).info(f"Используется быстрый промпт: {user_prompt[:50]}...")
        return user_prompt

    def _get_current_time_info(self) -> tuple:
        """Получает текущую дату и день недели"""
        from datetime import datetime

        now = datetime.now()
        today_date = now.strftime("%Y-%m-%d")
        today_weekday = WEEKDAY_MAP[now.strftime("%A")]
        today_lunar = cnlunar.Lunar(now, godType="8char")
        lunar_date = "%sвозраст%s%s\n" % (
            today_lunar.lunarYearCn,
            today_lunar.lunarMonthCn[:-1],
            today_lunar.lunarDayCn,
        )

        return today_date, today_weekday, lunar_date

    def _get_location_info(self, client_ip: str) -> str:
        """Получает информацию о местоположении по IP"""
        try:
            # Сначала проверяем кэш
            cached_location = self.cache_manager.get(self.CacheType.LOCATION, client_ip)
            if cached_location is not None:
                return cached_location

            # Если нет в кэше — получаем через API
            from core.utils.util import get_ip_info

            ip_info = get_ip_info(client_ip, self.logger)
            city = ip_info.get("city", "неизвестно")
            location = f"{city}"

            # Сохраняем в кэш
            self.cache_manager.set(self.CacheType.LOCATION, client_ip, location)
            return location
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Не удалось получить местоположение: {e}")
            return "неизвестно"

    def _get_weather_info(self, conn, location: str) -> str:
        """Получает информацию о погоде"""
        try:
            # Сначала проверяем кэш
            cached_weather = self.cache_manager.get(self.CacheType.WEATHER, location)
            if cached_weather is not None:
                return cached_weather

            # Если нет в кэше — вызываем функцию получения погоды
            from plugins_func.functions.get_weather import get_weather
            from plugins_func.register import ActionResponse

            # Вызов функции get_weather
            result = get_weather(conn, location=location, lang="zh_CN")
            if isinstance(result, ActionResponse):
                weather_report = result.result
                self.cache_manager.set(self.CacheType.WEATHER, location, weather_report)
                return weather_report
            return "Не удалось получить информацию о погоде"

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Ошибка получения погоды: {e}")
            return "Не удалось получить информацию о погоде"

    def update_context_info(self, conn, client_ip: str):
        """Обновляет контекстную информацию: местоположение и погоду"""
        try:
            # Получаем местоположение (из кэша или API)
            local_address = self._get_location_info(client_ip)
            # Получаем погоду (из кэша или API)
            self._get_weather_info(conn, local_address)
            self.logger.bind(tag=TAG).info(f"Контекстная информация обновлена")

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Ошибка обновления контекстной информации: {e}")

    def build_enhanced_prompt(
        self, user_prompt: str, device_id: str, client_ip: str = None
    ) -> str:
        """Формирует расширенный системный промпт с контекстом"""
        if not self.base_prompt_template:
            return user_prompt

        try:
            # Получаем актуальную информацию о времени
            today_date, today_weekday, lunar_date = self._get_current_time_info()

            # Получаем кэшированную информацию
            local_address = ""
            weather_info = ""

            if client_ip:
                # Местоположение из кэша
                local_address = (
                    self.cache_manager.get(self.CacheType.LOCATION, client_ip) or "неизвестно"
                )

                # Погода из кэша
                if local_address:
                    weather_info = (
                        self.cache_manager.get(self.CacheType.WEATHER, local_address)
                        or "неизвестно"
                    )

            # Заполняем шаблон
            template = Template(self.base_prompt_template)
            enhanced_prompt = template.render(
                base_prompt=user_prompt,
                current_time="{{current_time}}",
                today_date=today_date,
                today_weekday=today_weekday,
                lunar_date=lunar_date,
                local_address=local_address,
                weather_info=weather_info,
                emojiList=EMOJI_List,
            )
            device_cache_key = f"device_prompt:{device_id}"
            self.cache_manager.set(
                self.CacheType.DEVICE_PROMPT, device_cache_key, enhanced_prompt
            )
            self.logger.bind(tag=TAG).info(
                f"Расширенный промпт успешно сформирован, длина: {len(enhanced_prompt)}"
            )
            return enhanced_prompt

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Ошибка при формировании расширенного промпта: {e}")
            return user_prompt