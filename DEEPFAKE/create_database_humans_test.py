
from duckduckgo_search import ddg_images
from fastcore.all import *
from fastai.vision.all import *

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')

dest = Path("img_test")

dest.mkdir(exist_ok=True, parents=True)
download_images(dest, urls=search_images("human faces", max_images=30))
download_images(dest, urls=search_images("man standing up", max_images=30))
download_images(dest, urls=search_images("man standing woman standing up", max_images=30))
resize_images(dest, max_size=400, dest=dest)

failed = verify_images(get_image_files(dest))
failed.map(Path.unlink)
