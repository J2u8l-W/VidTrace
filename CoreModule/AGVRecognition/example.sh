python train.py  --train    --traindataset Video_Fusion --lr 1e-6
python train.py  --test  --load_path save/Video_Fusion/model/best_model.pt --testdataset Aphantasia
python train.py  --test  --load_path save/Video_Fusion/model/best_model.pt --testdataset ModelZero 
python train.py  --test  --load_path save/Video_Fusion/model/best_model.pt --testdataset Video_Fusion 
python train.py  --test  --load_path save/Video_Fusion/model/best_model.pt --testdataset T2VSynthesis 
python train.py  --test  --load_path save/Video_Fusion/model/best_model.pt --testdataset Tune-a-Video 


python train.py  --train    --traindataset Aphantasia --lr 1e-6
python train.py  --test  --load_path save/Aphantasia/model/best_model.pt --testdataset Aphantasia
python train.py  --test  --load_path save/Aphantasia/model/best_model.pt --testdataset ModelZero 
python train.py  --test  --load_path save/Aphantasia/model/best_model.pt --testdataset Video_Fusion 
python train.py  --test  --load_path save/Aphantasia/model/best_model.pt --testdataset T2VSynthesis 
python train.py  --test  --load_path save/Aphantasia/model/best_model.pt --testdataset Tune-a-Video 

python train.py  --train    --traindataset ModelZero --lr 1e-6
python train.py  --test  --load_path save/ModelZero/model/best_model.pt --testdataset Aphantasia
python train.py  --test  --load_path save/ModelZero/model/best_model.pt --testdataset ModelZero 
python train.py  --test  --load_path save/ModelZero/model/best_model.pt --testdataset Video_Fusion 
python train.py  --test  --load_path save/ModelZero/model/best_model.pt --testdataset T2VSynthesis 
python train.py  --test  --load_path save/ModelZero/model/best_model.pt --testdataset Tune-a-Video 

python train.py  --train    --traindataset T2VSynthesis --lr 1e-6
python train.py  --test  --load_path save/T2VSynthesis/model/best_model.pt --testdataset Aphantasia
python train.py  --test  --load_path save/T2VSynthesis/model/best_model.pt --testdataset ModelZero 
python train.py  --test  --load_path save/T2VSynthesis/model/best_model.pt --testdataset Video_Fusion 
python train.py  --test  --load_path save/T2VSynthesis/model/best_model.pt --testdataset T2VSynthesis 
python train.py  --test  --load_path save/T2VSynthesis/model/best_model.pt --testdataset Tune-a-Video 

python train.py  --train    --traindataset Tune-a-Video  --lr 1e-6
python train.py  --test  --load_path save/Tune-a-Video/model/best_model.pt --testdataset Aphantasia
python train.py  --test  --load_path save/Tune-a-Video/model/best_model.pt --testdataset ModelZero 
python train.py  --test  --load_path save/Tune-a-Video/model/best_model.pt --testdataset Video_Fusion 
python train.py  --test  --load_path save/Tune-a-Video/model/best_model.pt --testdataset T2VSynthesis 
python train.py  --test  --load_path save/Tune-a-Video/model/best_model.pt --testdataset Tune-a-Video 