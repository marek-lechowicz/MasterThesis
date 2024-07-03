'''This script is used to extract features from a single image using the Radiomics library. It is used for debugging purposes.'''
from radiomics import featureextractor

# param path
param = "preparation_and_extraction/params/extraction_params.yaml"
img_path = "data/artwork/RGB/red/train_and_test/FAKE/R_0.jpg"
mask_path = "data/masks/mask512x512.jpg"

# create feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor(param)

# disable all image types
extractor.disableAllImageTypes()

# enable LoG filter with sigma 3.0
extractor.enableImageTypeByName('LoG', customArgs={'sigma': [3.0]})

print(extractor.enabledImagetypes)

# extract features
result = extractor.execute(img_path, mask_path)
# print(result)


print("ALLOHA")