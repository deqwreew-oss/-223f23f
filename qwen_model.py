"""
Qwen Image Generation Model Module
Оптимизированный модуль для работы с моделью Qwen через FAL API
"""

import os
import re
import logging
from typing import Any, Dict, List, Union
from dataclasses import dataclass

import fal_client


# Настройка логгера для модуля
logger = logging.getLogger("qwen-model")


# Типы для размеров изображений
ImageSizeValue = Union[str, Dict[str, int]]

# Упрощенные размеры с понятными названиями
SIZE_MAP = {
    "Квадрат 2K": {"width": 2048, "height": 2048},
    "Портрет 9:16": "portrait_16_9", 
    "Альбом 16:9": "landscape_16_9",
    "Альбом 4:3": "landscape_4_3",
}

ALLOWED_SIZES: List[str] = list(SIZE_MAP.keys())


@dataclass
class QwenGenerationOptions:
    """Настройки генерации для модели Qwen"""
    image_size: str = "landscape_4_3"
    num_inference_steps: int = 30
    guidance_scale: float = 2.5
    num_images: int = 1
    enable_safety_checker: bool = True
    output_format: str = "png"
    negative_prompt: str = " "
    acceleration: str = "none"
    seed: Union[int, None] = None


@dataclass
class GenerationResult:
    """Результат генерации изображения"""
    success: bool
    images: List[Dict[str, Any]]
    error: Union[str, None] = None
    
    @property
    def image_urls(self) -> List[str]:
        """Возвращает список URL изображений"""
        if not self.success or not self.images:
            return []
        
        urls = []
        for img in self.images:
            if isinstance(img, dict) and img.get("url"):
                urls.append(img["url"])
        return urls


class QwenImageGenerator:
    """Класс для работы с моделью Qwen Image через FAL API"""
    
    MODEL_ENDPOINT = "fal-ai/qwen-image"
    MODEL_NAME = "Qwen-Image"
    
    def __init__(self, fal_api_key: str = None):
        """
        Инициализация генератора
        
        Args:
            fal_api_key: API ключ для FAL, если None - берется из переменной окружения
        """
        self.fal_api_key = fal_api_key or os.getenv("FAL_KEY")
        if not self.fal_api_key:
            raise ValueError("FAL_KEY не установлен. Укажите API ключ или установите переменную окружения.")
    
    @staticmethod
    def parse_image_size(value: str) -> ImageSizeValue:
        """
        Парсит размер изображения из строки
        
        Args:
            value: строка с размером (например "1024x576" или "square_hd")
            
        Returns:
            Размер в формате API или словарь с width/height
        """
        v = value.strip().lower()
        
        # Проверяем формат widthxheight (например, 1024x576)
        match = re.match(r"^(\d{2,5})x(\d{2,5})$", v)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return {"width": width, "height": height}

        # Проверяем предустановленные размеры API
        allowed_api_sizes = {
            "square_hd", "square", "portrait_4_3", "portrait_16_9",
            "landscape_4_3", "landscape_16_9"
        }
        
        if v in allowed_api_sizes:
            return v
            
        # По умолчанию возвращаем альбомный формат
        return "landscape_4_3"
    
    @staticmethod
    def parse_flags_and_prompt(text: str) -> tuple[str, Dict[str, Any]]:
        """
        Парсит флаги и промпт из текста команды
        
        Args:
            text: текст команды с возможными флагами
            
        Returns:
            Кортеж (промпт, словарь_опций)
        """
        # Разбиваем текст с учетом кавычек
        try:
            import shlex
            tokens = shlex.split(text)
        except Exception:
            tokens = text.split()

        options: Dict[str, Any] = {}
        prompt_tokens: List[str] = []

        for token in tokens:
            if token.startswith("--") and "=" in token:
                key, value = token[2:].split("=", 1)
                key = key.strip().lower()
                value = value.strip().strip('"').strip("'")

                if key in {"size", "image_size"}:
                    options["image_size"] = QwenImageGenerator.parse_image_size(value)
                elif key in {"steps", "num_inference_steps"}:
                    try:
                        options["num_inference_steps"] = max(1, int(value))
                    except ValueError:
                        pass
                elif key in {"guidance", "guidance_scale"}:
                    try:
                        options["guidance_scale"] = float(value)
                    except ValueError:
                        pass
                elif key in {"seed"}:
                    try:
                        options["seed"] = int(value)
                    except ValueError:
                        pass
                elif key in {"images", "num_images"}:
                    try:
                        # Ограничиваем от 1 до 4 изображений
                        options["num_images"] = max(1, min(4, int(value)))
                    except ValueError:
                        pass
                elif key in {"format", "output_format"}:
                    fmt = value.lower()
                    if fmt in {"png", "jpeg"}:
                        options["output_format"] = fmt
                elif key in {"negative", "negative_prompt"}:
                    options["negative_prompt"] = value
                elif key in {"acceleration"}:
                    acc = value.lower()
                    if acc in {"none", "regular", "high"}:
                        options["acceleration"] = acc
                elif key in {"safety", "enable_safety_checker"}:
                    options["enable_safety_checker"] = value.lower() in {"1", "true", "yes", "on"}
            else:
                prompt_tokens.append(token)

        prompt = " ".join(prompt_tokens).strip()
        return prompt, options
    
    def generate_image(self, prompt: str, options: QwenGenerationOptions = None) -> GenerationResult:
        """
        Генерирует изображение по промпту (синхронная функция)
        
        Args:
            prompt: текстовое описание изображения
            options: настройки генерации
            
        Returns:
            Результат генерации
        """
        if not prompt.strip():
            return GenerationResult(success=False, images=[], error="Промпт не может быть пустым")
        
        if options is None:
            options = QwenGenerationOptions()
        
        # Подготавливаем аргументы для API
        arguments = {
            "prompt": prompt,
            "image_size": options.image_size,
            "num_inference_steps": options.num_inference_steps,
            "guidance_scale": options.guidance_scale,
            "num_images": options.num_images,
            "enable_safety_checker": options.enable_safety_checker,
            "output_format": options.output_format,
            "negative_prompt": options.negative_prompt,
            "acceleration": options.acceleration,
        }
        
        if options.seed is not None:
            arguments["seed"] = options.seed

        # Логируем начало генерации
        logger.info("🎨 Starting Qwen generation | Size: %s | Steps: %s | Safety: %s", 
                    arguments.get("image_size", "unknown"),
                    arguments.get("num_inference_steps", "unknown"),
                    "ON" if arguments.get("enable_safety_checker", True) else "OFF")

        def on_queue_update(update: Any) -> None:
            # Подавляем внутренние логи FAL API
            pass

        try:
            result: Dict[str, Any] = fal_client.subscribe(
                self.MODEL_ENDPOINT,
                arguments=arguments,
                with_logs=False,
                on_queue_update=on_queue_update,
            )
            
            if not isinstance(result, dict):
                error_msg = f"Invalid response format: {type(result)}"
                logger.error("❌ Qwen API error | %s", error_msg)
                return GenerationResult(success=False, images=[], error=error_msg)
            
            images = result.get("images", [])
            if not images:
                # Возвращаем единый текст ошибки для простоты обработки наверху
                return GenerationResult(success=False, images=[], error="no_images")
            
            # Логируем успешную генерацию
            images_count = len(images)
            logger.info("✅ Qwen generation completed | Images: %d", images_count)
            
            return GenerationResult(success=True, images=images)
            
        except Exception as e:
            # Улучшенная обработка ошибок FAL API
            error_msg = QwenImageGenerator._parse_error_message(e)
            logger.error("❌ Qwen API failed | %s", error_msg)
            return GenerationResult(success=False, images=[], error=error_msg)
    
    def generate_from_command_text(self, command_text: str) -> GenerationResult:
        """
        Генерирует изображение из текста команды с флагами
        
        Args:
            command_text: текст команды (например "/imagine кот --size=square_hd")
            
        Returns:
            Результат генерации
        """
        prompt, parsed_options = self.parse_flags_and_prompt(command_text)
        
        if not prompt:
            return GenerationResult(success=False, images=[], error="Промпт пуст")
        
        # Создаем настройки генерации из парсенных опций
        options = QwenGenerationOptions()
        
        # Применяем парсенные опции
        if "image_size" in parsed_options:
            options.image_size = parsed_options["image_size"]
        if "num_inference_steps" in parsed_options:
            options.num_inference_steps = parsed_options["num_inference_steps"]
        if "guidance_scale" in parsed_options:
            options.guidance_scale = parsed_options["guidance_scale"]
        if "num_images" in parsed_options:
            options.num_images = parsed_options["num_images"]
        if "enable_safety_checker" in parsed_options:
            options.enable_safety_checker = parsed_options["enable_safety_checker"]
        if "output_format" in parsed_options:
            options.output_format = parsed_options["output_format"]
        if "negative_prompt" in parsed_options:
            options.negative_prompt = parsed_options["negative_prompt"]
        if "acceleration" in parsed_options:
            options.acceleration = parsed_options["acceleration"]
        if "seed" in parsed_options:
            options.seed = parsed_options["seed"]
        
        return self.generate_image(prompt, options)
    
    @staticmethod
    def convert_user_size_to_api(user_size: str) -> str:
        """
        Конвертирует пользовательский размер в API формат
        
        Args:
            user_size: размер в понятном формате (например "Квадрат")
            
        Returns:
            Размер в API формате
        """
        return SIZE_MAP.get(user_size, "landscape_4_3")
    
    @staticmethod
    def _parse_error_message(error: Exception) -> str:
        """
        Парсит сообщение об ошибке от FAL API
        
        Args:
            error: исключение от FAL API
            
        Returns:
            Читаемое сообщение об ошибке
        """
        error_str = str(error)
        
        # Проверяем на content policy violation
        if "content_policy_violation" in error_str.lower():
            return "content_policy_violation"
        
        # Пытаемся распарсить JSON ошибку если это список
        try:
            # Ищем JSON в строке ошибки
            import json
            import re
            
            # Ищем JSON массив в строке
            json_match = re.search(r'\[{.*}\]', error_str)
            if json_match:
                error_list = json.loads(json_match.group())
                if isinstance(error_list, list) and len(error_list) > 0:
                    first_error = error_list[0]
                    if isinstance(first_error, dict):
                        # Возвращаем тип ошибки если есть
                        error_type = first_error.get("type", "")
                        error_msg = first_error.get("msg", "")
                        
                        if error_type == "content_policy_violation":
                            return "content_policy_violation"
                        elif error_msg:
                            return error_msg
                        elif error_type:
                            return error_type
            
        except (json.JSONDecodeError, ValueError, AttributeError):
            # Если не удалось распарсить JSON, возвращаем исходную ошибку
            pass
        
        return error_str


# Утилитарные функции для обратной совместимости
def create_qwen_generator() -> QwenImageGenerator:
    """Создает экземпляр генератора Qwen"""
    return QwenImageGenerator()


def generate_qwen_image(prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Генерирует изображение через Qwen (для обратной совместимости)
    
    Args:
        prompt: промпт для генерации
        options: словарь с настройками
        
    Returns:
        Словарь с результатом в старом формате
    """
    try:
        generator = create_qwen_generator()
        
        # Конвертируем старый формат опций в новый
        qwen_options = QwenGenerationOptions()
        if options:
            if "image_size" in options:
                qwen_options.image_size = options["image_size"]
            if "num_inference_steps" in options:
                qwen_options.num_inference_steps = options["num_inference_steps"]
            if "guidance_scale" in options:
                qwen_options.guidance_scale = options["guidance_scale"]
            if "num_images" in options:
                qwen_options.num_images = options["num_images"]
            if "enable_safety_checker" in options:
                qwen_options.enable_safety_checker = options["enable_safety_checker"]
            if "output_format" in options:
                qwen_options.output_format = options["output_format"]
            if "negative_prompt" in options:
                qwen_options.negative_prompt = options["negative_prompt"]
            if "acceleration" in options:
                qwen_options.acceleration = options["acceleration"]
            if "seed" in options:
                qwen_options.seed = options["seed"]
        
        result = generator.generate_image(prompt, qwen_options)
        
        # Возвращаем в старом формате для совместимости
        if result.success:
            return {"images": result.images}
        else:
            return {"images": [], "error": result.error}
            
    except Exception as e:
        # Используем улучшенную обработку ошибок
        error_msg = QwenImageGenerator._parse_error_message(e)
        return {"images": [], "error": error_msg}

