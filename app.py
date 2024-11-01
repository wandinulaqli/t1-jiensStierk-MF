from flask import Flask,render_template,url_for,request, redirect
import pandas as pd
import numpy as np
import random
import sklearn.model_selection as ms
import sklearn.metrics as met
import joblib

app = Flask(__name__)
model = joblib.load('model.sav')

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        # prediksiJenisStiker
        df = pd.read_csv('dataset.csv')
        # import sklearn.model_selection as ms
        X = df[['halus','tebal','transparan','berlubang','tahanAir','memantulkanCahaya']]
        y = df['jenis']
        # splitDataTrain
        rInt = random.randint(10,99)
        # print(r.type)
        X_train,X_test,y_train,y_test = ms.train_test_split(X,y, test_size=rInt, random_state=0)
        # scoringModel_hasilPredict
        y_prediksi=model.predict(X_test) # model ML

        print('\n', len(y_prediksi))
        score = model.score(X_test, y_test) # ??scoreModel_bawaan
        print('score :', score)
        # FinishModel
        print('\n',model)
        print('==>','X_train:',len(X_train),'|| X_test:',len(X_test), '|| y_train:',len(y_train),'|| y_test',len(y_test))
        print('\n')
        X_test['Labels'] = y_prediksi
        print(X_test)
        # classification_report[Model]
        print('\n',model)
        confusionmatrix = met.confusion_matrix(y_test, y_prediksi)
        print(confusionmatrix)
        scoreMet = met.accuracy_score (y_test, y_prediksi)
        print('==> Accuracy:',scoreMet)
        report = met.classification_report(y_test, y_prediksi)# reportModel
        print('==> Report:')
        print(report)
        print('')

        # print(request.form.get('name1'))
        input1 = request.form.get('name1')
        # print(request.form.get('name2'))
        input2 = request.form.get('name2')
        # halus,tebal,transparan,berlubang,tahanAir,memantulkanCahaya,jenis
        # 1,0,0,0,1,0,vinyl
        # 0,1,0,0,1,0,karbonKevlar
        # 0,1,0,0,1,1,fosfor
        # 1,0,1,0,1,0,transparan
        # 1,0,0,0,1,0,oracal
        # 1,0,0,0,1,0,scotLight
        # 0,0,0,1,1,0,oneWayVision
        # 1,0,0,0,0,0,choromo
        # 1,0,0,0,1,0,sandBlast
        # 1,0,0,0,0,0,hvs
        # npArray predict
        rInt2=random.randint(0,1)
        print(rInt2)
        if (input1=='halus')and((input2=='tahanAir')):
            npX = np.array([[1,0,0,0,1,0]])
            # vinyl, oracal, scotLight, choromo, sandBlast

        elif (input1=='tebal')and(input2=='memantulkanCahaya'):
            # karbonKevlar
            npX = np.array([[0,1,0,0,1,1]])

        elif ((input1=='tebal')or(input1=='halus'))and(input2=='memantulkanCahaya'):
            # fosfor
            npX = np.array([[0,1,0,0,1,1]])

        elif (input1=='transparan')and((input2=='tahanAir')or(input2=='tahanSinarMatahari')):
            # transparan
            npX = np.array([[1,0,1,0,1,0]])

        elif (input1=='berlubang')and((input2=='tahanAir')or(input2=='tahanSinarMatahari')):
            # oneWayVision
            npX = np.array([[0,0,0,1,1,0]])

        elif ((input1=='halus')or(input1=='tebal'))and(input2=='dayaRekatKuat'):
            # hvs
            npX = np.array([[1,0,0,0,0,0]])
        else:
            npX = np.array([[rInt2,rInt2,rInt2,rInt2,rInt2,rInt2]])

        print(npX)
        # hasilPredict(npy)
        npy = model.predict(npX)

        # vinyl, oracal, scotLight, choromo, sandBlast
        if npy==['vinyl']:
            g='https://images.tokopedia.net/img/cache/500-square/VqbcmM/2021/9/5/1c622191-dbb5-41e7-8c46-43bd788db52b.jpg?ect=4g'
        elif npy==['oracal']:
            g='https://images.tokopedia.net/img/cache/500-square/product-1/2020/5/2/9817853/9817853_e290daf0-6b03-4e2d-9133-4b049c5a5eb1_1418_1418?ect=4g'
        elif npy==['scotLight']:
            g='https://ecs7.tokopedia.net/img/cache/700/product-1/2019/4/4/421746969/421746969_6a6ce208-47b3-423b-baed-b4fa1a71a65f_700_700.jpg'
        elif npy==['choromo']:
            g='https://images.tokopedia.net/img/cache/500-square/VqbcmM/2022/8/22/bc46ab2c-240c-4acb-81e9-6078cb7c81a6.jpg?ect=4g'
        elif npy==['sandBlast']:
            g='https://images.tokopedia.net/img/cache/500-square/VqbcmM/2022/6/13/ee2b9b51-65f9-49cc-be9b-0eb873f8ca71.jpg'
        elif npy==['karbonKevlar']:
            g='https://images.tokopedia.net/img/cache/500-square/VqbcmM/2022/10/26/3a538669-ba20-4793-b470-5c63d1ddb94e.jpg'
        elif npy==['fosfor']:
            g='https://images.tokopedia.net/img/cache/500-square/product-1/2018/8/7/403579/403579_febb3ecb-e116-4a15-8953-84e66f123249_700_700.jpg'
        elif npy==['transparan']:
            g='https://ecs7.tokopedia.net/img/cache/700/product-1/2019/11/6/1534553/1534553_25285733-f517-4cef-9621-2c40940d1d2f_1183_1183'
        elif npy==['oneWayVision']:
            g='https://images.tokopedia.net/img/cache/500-square/VqbcmM/2022/5/23/b49ac6e1-14cd-49d9-98b6-ce94dba5d592.png'
        elif npy==['hvs']:
            g='https://images.tokopedia.net/img/cache/500-square/VqbcmM/2022/12/25/547eb81a-a2ba-4949-bbab-adab0a2de2e1.jpg'

        return render_template('index.html', hpredict=npy, gambar=g, ac=round(scoreMet, 2))

# run app
if __name__ == '__main__':
    app.run(debug=True)
