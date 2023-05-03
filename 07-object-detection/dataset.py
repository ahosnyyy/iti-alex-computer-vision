

from bing_image_downloader import downloader

downloader.download("dogs", limit=20, output_dir="./", adult_filter_off=True, force_replace=False, timeout=30)