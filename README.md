# laTeXuTiL
Repository that contains python methods to automate some laTeX tasks

## Getting `.summary()` output into laTeX

Works on all types of tf.keras.Model (sequential, functionnal, self defined ..)

```
from tf_model_utils import tf2Mod2TeX

% Define `model`

txtSum = tf2Mod2TeX(model, 
                    modelName ='model',
                    modelLabel = 'm1',
                    char_dbl='=', lay_line_sep='\\hline')

f = open('namefile.txt', 'w')
f.write(txtSum)
f.close()
```
