
from duckduckgo_search import ddg_images
from fastcore.all import *
from fastai.vision.all import *

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')

dest = Path("D_Trump_Real")

dest.mkdir(exist_ok=True, parents=True)
download_images(dest, urls=search_images("Donald Trump photo", max_images=100))
download_images(dest, urls=search_images("Donald Trump smiling photo", max_images=100))
download_images(dest, urls=search_images("Donald Trump speaking photo", max_images=100))
download_images(dest, urls=search_images("Donald Trump raising hand photo", max_images=100))
download_images(dest, urls=search_images("Donald Trump black and white photo", max_images=100))
resize_images(dest, max_size=400, dest=dest)

failed = verify_images(get_image_files(dest))
failed.map(Path.unlink)
