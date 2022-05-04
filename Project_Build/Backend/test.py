from signature_detect.loader import Loader


loader = Loader(low_threshold=(0, 0, 250), high_threshold=(255, 255, 255))


masks = loader.get_masks("signature.png")
print(masks)