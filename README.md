LSTM Programming Teacher
========================

check the code is valid or not without running it

Training
--------

```
python bin/atcoder_crowler.py abc041 a gpp_5.3.0 WA
python bin/atcoder_crowler.py abc041 a gpp_5.3.0 AC
python bin/train_model.py abc041 a gpp_5.3.0 --batch=1 --max_length=200 --epoch=1000 --testnum=10 --augmentation=5 --gpu=-1
```

Check The Code
--------------

```
python bin/programming_advisor.py output/1468914085.51/70_model.npz output/1468914085.51/70_chara_encoder.dump   ./data/abc041/a/gpp_5.3.0/dummy/788889
```
