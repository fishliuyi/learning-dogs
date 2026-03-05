import requests
from PIL import Image, ImageFile
from io import BytesIO
from pathlib import Path
from typing import Union, Optional, Tuple
import time

# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageLoader:
    """优化的图像加载器"""

    def __init__(self, timeout: int = 30, max_retries: int = 3,
                 chunk_size: int = 8192, max_size: int = 50 * 1024 * 1024):
        """
        初始化图像加载器

        Args:
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            chunk_size: 下载块大小
            max_size: 最大文件大小（字节）
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.max_size = max_size

        # 默认请求头
        self.default_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }

    def load_from_url(self, url: str,
                      headers: Optional[dict] = None,
                      verify_ssl: bool = True) -> Image.Image:
        """
        从URL安全加载图像

        Args:
            url: 图像URL
            headers: 自定义请求头
            verify_ssl: 是否验证SSL证书

        Returns:
            PIL Image对象

        Raises:
            ValueError: URL无效或图像格式不支持
            ConnectionError: 网络连接失败
            IOError: 图像加载失败
        """
        if not self._is_valid_url(url):
            raise ValueError(f"无效的URL: {url}")

        # 合并请求头
        request_headers = {**self.default_headers}
        if headers:
            request_headers.update(headers)

        # 重试机制
        for attempt in range(self.max_retries):
            try:
                image = self._download_image(url, request_headers, verify_ssl)
                self._validate_image(image, url)
                return image

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise ConnectionError(f"下载图像失败 ({url}): {str(e)}")
                time.sleep(2 ** attempt)  # 指数退避

    def load_from_path(self, path: Union[str, Path]) -> Image.Image:
        """
        从本地路径加载图像

        Args:
            path: 图像文件路径

        Returns:
            PIL Image对象

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 路径无效
            IOError: 图像加载失败
        """
        path_obj = Path(path).resolve()

        if not path_obj.exists():
            raise FileNotFoundError(f"图像文件不存在: {path}")

        if not path_obj.is_file():
            raise ValueError(f"路径不是文件: {path}")

        try:
            image = Image.open(path_obj)
            self._validate_image(image, str(path))
            return image.convert('RGB')
        except Exception as e:
            raise IOError(f"加载图像文件失败 ({path}): {str(e)}")

    def _is_valid_url(self, url: str) -> bool:
        """验证URL有效性"""
        if not isinstance(url, str) or not url.strip():
            return False
        return url.startswith(('http://', 'https://'))

    def _download_image(self, url: str, headers: dict, verify_ssl: bool) -> Image.Image:
        """下载图像数据"""
        response = requests.get(
            url,
            headers=headers,
            timeout=self.timeout,
            stream=True,
            verify=verify_ssl
        )
        response.raise_for_status()

        # 检查内容长度
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > self.max_size:
            raise ValueError(
                f"图像文件过大: {int(content_length) / 1024 / 1024:.1f}MB")

        # 分块读取数据
        image_data = BytesIO()
        downloaded_size = 0

        for chunk in response.iter_content(chunk_size=self.chunk_size):
            if chunk:
                downloaded_size += len(chunk)
                if downloaded_size > self.max_size:
                    raise ValueError(
                        f"图像文件超过大小限制: {self.max_size / 1024 / 1024:.1f}MB")
                image_data.write(chunk)

        image_data.seek(0)
        return Image.open(image_data).convert('RGB')

    def _validate_image(self, image: Image.Image, source: str):
        """验证图像有效性"""
        if image is None:
            raise IOError(f"无法加载图像: {source}")

        # 检查图像尺寸
        width, height = image.size
        if width < 10 or height < 10:
            raise ValueError(f"图像尺寸过小 {width}x{height}: {source}")

        if width > 10000 or height > 10000:
            raise ValueError(f"图像尺寸过大 {width}x{height}: {source}")

    def get_image_info(self, image: Image.Image) -> dict:
        """获取图像基本信息"""
        return {
            'size': image.size,
            'mode': image.mode,
            'format': image.format,
            'bands': image.getbands()
        }


class ImageProcessor:
    """图像处理器 - 提供常用的图像处理功能"""

    @staticmethod
    def resize_with_aspect_ratio(image: Image.Image,
                                 target_size: Tuple[int, int],
                                 maintain_aspect: bool = True) -> Image.Image:
        """保持宽高比调整图像大小"""
        if maintain_aspect:
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
            return image
        else:
            return image.resize(target_size, Image.Resampling.LANCZOS)

    @staticmethod
    def validate_image_format(filepath: Union[str, Path]) -> bool:
        """验证图像文件格式"""
        try:
            with Image.open(filepath) as img:
                img.verify()
            return True
        except Exception:
            return False


# 便利函数
def load_image(source: Union[str, Path], **kwargs) -> Image.Image:
    """
    通用图像加载函数

    Args:
        source: URL或文件路径
        **kwargs: 传递给ImageLoader的参数

    Returns:
        PIL Image对象
    """
    loader = ImageLoader(**kwargs)

    if isinstance(source, (str, Path)) and str(source).startswith(('http://', 'https://')):
        return loader.load_from_url(str(source))
    else:
        return loader.load_from_path(source)


def quick_load_image(source: str) -> Image.Image:
    """快速加载图像的便利函数"""
    return load_image(source, timeout=10, max_retries=1)
