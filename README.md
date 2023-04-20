# laTeXuTiL
Repository that contains python methods to automate some laTeX tasks

## Getting `.summary()` output into laTeX

```
from tf_model_utils import tf2Mod2TeX

txtSum = tf2Mod2TeX(spectral_predictor.model.model, 
                    modelName = modelName + ' for IRIS data',
                    modelLabel = modelName,
                    col_keys=['Layer (type)', 'Output Shape', 
                              "Param #", 'Connected to'], 
                    char_dbl='=', lay_line_sep='\\hline')

f = open('namefile.txt', 'w')
f.write(txtSum)
f.close()
```
