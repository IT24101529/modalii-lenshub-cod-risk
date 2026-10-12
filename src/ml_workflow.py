"""Reproducible COD risk study. Selection uses training folds, not holdout metrics.

Run: python src/run_project.py --dataset /path/to/approved/dataset.csv
The dataset-preparing member reports matching DeliveryOrdersReport.csv and
ReturnOrdersReport.csv on Waybill Number: returns are 1, successful/non-returned
deliveries are 0. This replaces the initial proposed collection-based target.
Original exports and merge code are unavailable for independent verification.
All 206 zero-COD orders are retained and flagged; their business meaning is unknown.
"""
from pathlib import Path
import hashlib, json, platform, time, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve)
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

SEED = 20261011
NUMERIC = ['COD', 'Delivery_Charge_Ratio']
CATEGORICAL = ['Is_Outstation', 'COD_Is_Zero']
FEATURES = NUMERIC + CATEGORICAL

class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Rebuild deterministic features after synthetic primitive COD/route samples."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        a=np.asarray(X)
        cod=a[:,0];route=a[:,1]
        charge=np.where(route==1,425.,375.)
        ratio=np.divide(charge,cod,out=np.zeros(len(cod)),where=cod!=0)
        return pd.DataFrame({'COD':cod,'Delivery_Charge_Ratio':ratio,
            'Is_Outstation':route,'COD_Is_Zero':(cod==0).astype(int)})[FEATURES]

def locate_root():
    here = Path(__file__).resolve()
    return here.parents[1]

def load_data(root=None, dataset=None):
    root = Path(root) if root else locate_root()
    path = Path(dataset).expanduser().resolve() if dataset else root/'data/Modalii_Lenshub_ML_Dataset_Final.csv'
    df = pd.read_csv(path)
    assert df.Order_Record_ID.is_unique
    assert set(df.Is_Returned.unique()) == {0,1}
    assert (df.COD >= 0).all()
    df['COD_Is_Zero'] = (df.COD == 0).astype(int)
    # Recompute the documented ratio from dispatch information; zero is an
    # explicit sentinel, accompanied by COD_Is_Zero (not an economic ratio).
    df['Delivery_Charge_Ratio'] = np.divide(df['Delivery Charge'],df.COD,
        out=np.zeros(len(df)),where=df.COD.to_numpy()!=0)
    return df

def audit_data(df):
    return {'rows':len(df), 'returned':int(df.Is_Returned.sum()),
      'prevalence':float(df.Is_Returned.mean()), 'missing_cells':int(df.isna().sum().sum()),
      'duplicate_record_ids':int(df.Order_Record_ID.duplicated().sum()),
      'duplicate_nonidentifier_rows':int(df.drop(columns='Order_Record_ID').duplicated().sum()),
      'weight_unique':int(df.Weight.nunique()), 'zero_cod':int(df.COD.eq(0).sum()),
      'zero_cod_returned':int(df.loc[df.COD.eq(0),'Is_Returned'].sum()),
      'route_charge':pd.crosstab(df['Delivery Charge'],df.Is_Outstation).to_dict()}

def models(y):
    numeric = Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    pre = ColumnTransformer([('numeric',numeric,NUMERIC),
        ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATEGORICAL)],sparse_threshold=0)
    ratio = float((y==0).sum()/(y==1).sum())
    estimators = {
      'Dummy baseline':DummyClassifier(strategy='most_frequent'),
      'Logistic regression':LogisticRegression(class_weight='balanced',max_iter=2000,random_state=SEED),
      'Decision tree':DecisionTreeClassifier(max_depth=5,min_samples_leaf=20,class_weight='balanced',random_state=SEED),
      'Random forest':RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=10,class_weight='balanced',random_state=SEED,n_jobs=2),
      'Hist gradient boosting':HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1,class_weight='balanced',random_state=SEED),
      'XGBoost':XGBClassifier(n_estimators=200,max_depth=3,learning_rate=.05,subsample=.8,colsample_bytree=.8,scale_pos_weight=ratio,eval_metric='logloss',random_state=SEED,n_jobs=2)
    }
    result={k:Pipeline([('preprocess',clone(pre)),('model',v)]) for k,v in estimators.items()}
    # Resample primitive COD and categorical route only. Recompute the ratio
    # and zero flag afterwards; interpolating derived ratios breaks their identity.
    # Only one numeric dimension is resampled, so relative numeric scaling is moot.
    pre_smote=ColumnTransformer([('numeric',SimpleImputer(strategy='median'),['COD']),('category','passthrough',['Is_Outstation'])])
    result['Random forest with SMOTENC']=ImbPipeline([
      ('primitive',pre_smote),('resample',SMOTENC(categorical_features=[1],random_state=SEED)),
      ('derive',DerivedFeatures()),('preprocess',clone(pre)),
      ('model',RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=10,random_state=SEED,n_jobs=2))])
    return result

def metrics(y,score,threshold=.5):
    pred=(np.asarray(score)>=threshold).astype(int)
    return {'Accuracy':accuracy_score(y,pred),'Precision_Returned':precision_score(y,pred,zero_division=0),
      'Recall_Returned':recall_score(y,pred,zero_division=0),'F1_Returned':f1_score(y,pred,zero_division=0),
      'ROC_AUC':roc_auc_score(y,score),'Average_Precision':average_precision_score(y,score)}

def threshold_from_oof(y,score):
    precision,recall,threshold=precision_recall_curve(y,score)
    f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
    # Deterministic tie break: highest threshold, limiting unnecessary flags.
    ix=np.flatnonzero(np.isclose(f1,f1.max()))[-1]
    return float(threshold[ix])

def cross_validate(X,y):
    splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    oof={name:np.zeros(len(y)) for name in models(y)}
    foldrows=[]
    for name in oof:
        print('Cross validating:',name,flush=True)
        for fold,(train,valid) in enumerate(splitter.split(X,y),1):
            # Positive weights depend on this fold only.
            model=models(y.iloc[train])[name]
            model.fit(X.iloc[train],y.iloc[train])
            score=model.predict_proba(X.iloc[valid])[:,1]
            oof[name][valid]=score
            foldrows.append({'Model':name,'Fold':fold,**metrics(y.iloc[valid],score)})
    folds=pd.DataFrame(foldrows)
    rows=[]
    for name,score in oof.items():
        f=folds[folds.Model==name]
        rows.append({'Model':name,'Mean_AP':f.Average_Precision.mean(),'SD_AP':f.Average_Precision.std(ddof=1),
          'Mean_ROC_AUC':f.ROC_AUC.mean(),'Mean_F1_at_0_5':f.F1_Returned.mean(),
          'OOF_Threshold':.5 if name=='Dummy baseline' else threshold_from_oof(y,score),
          'OOF_AP':average_precision_score(y,score)})
    summary=pd.DataFrame(rows).sort_values(['Mean_AP','Model'],ascending=[False,True]).reset_index(drop=True)
    return summary,folds,oof

def bootstrap_ci(y,score,threshold):
    rng=np.random.default_rng(SEED)
    y=np.asarray(y); score=np.asarray(score)
    byclass=[np.flatnonzero(y==c) for c in [0,1]]
    values=[]
    for _ in range(1000):
        ix=np.concatenate([rng.choice(a,len(a),replace=True) for a in byclass])
        values.append(metrics(y[ix],score[ix],threshold))
    out=pd.DataFrame(values)
    return {c:[float(v) for v in out[c].quantile([.025,.975])] for c in out}

def make_figures(df,cv,holdout,y,score,threshold,out):
    plt.rcParams.update({'font.size':10,'figure.dpi':130})
    def save(name):
        plt.tight_layout(); plt.savefig(out/name,bbox_inches='tight'); plt.close()
    counts=df.Is_Returned.value_counts().sort_index()
    plt.figure(figsize=(7,3.2)); plt.bar(['Successful','Returned'],counts,color=['#315b79','#c46a3a'])
    plt.ylabel('Orders'); plt.title('Supplied delivery outcome labels')
    for i,v in enumerate(counts):plt.text(i,v+40,f'{v:,}',ha='center')
    save('01_Target_Class_Distribution.png')
    plt.figure(figsize=(7,3.2))
    df.boxplot(column='COD',by='Is_Returned',ax=plt.gca(),showfliers=False)
    plt.suptitle('');plt.title('COD amount by supplied outcome');plt.xlabel('Outcome 0 successful 1 returned');plt.ylabel('LKR')
    save('02_COD_Distribution_by_Outcome.png')
    rates=df.groupby('Is_Outstation').Is_Returned.mean()
    plt.figure(figsize=(7,3.2));plt.bar(['Lower charge route','Higher charge route'],rates*100,color='#315b79');plt.ylabel('Returned orders percent');plt.title('Descriptive route return rates')
    save('03_Return_Rate_by_Route.png')
    plt.figure(figsize=(8,4));plt.barh(cv.Model.iloc[::-1],cv.Mean_AP.iloc[::-1],xerr=cv.SD_AP.iloc[::-1],color='#315b79')
    plt.axvline(df.Is_Returned.mean(),color='#c46a3a',linestyle='--',label='Overall prevalence reference')
    plt.xlabel('Mean cross validation average precision');plt.legend(loc='lower left',bbox_to_anchor=(0,1.01));save('04_CV_Model_Comparison.png')
    fpr,tpr,_=roc_curve(y,score);plt.figure(figsize=(6,3.6));plt.plot(fpr,tpr,label='Selected model');plt.plot([0,1],[0,1],'--',color='gray');plt.xlabel('False positive rate');plt.ylabel('True positive rate');plt.title('Final holdout ROC curve');plt.legend();save('05_ROC_Curve.png')
    precision,recall,_=precision_recall_curve(y,score);plt.figure(figsize=(6,3.6));plt.plot(recall,precision,label='Selected model');plt.axhline(np.mean(y),linestyle='--',color='#c46a3a',label='Holdout prevalence');plt.xlabel('Recall');plt.ylabel('Precision');plt.title('Final holdout precision recall curve');plt.legend();save('06_Precision_Recall_Curve.png')
    cm=confusion_matrix(y,np.asarray(score)>=threshold)
    plt.figure(figsize=(5,3.5));plt.imshow(cm,cmap='Blues');plt.xticks([0,1],['Low risk','Review flag']);plt.yticks([0,1],['Successful','Returned']);plt.xlabel('Prediction');plt.ylabel('Actual outcome');plt.title('Selected threshold holdout confusion matrix')
    for (i,j),v in np.ndenumerate(cm):plt.text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black')
    save('07_Selected_Confusion_Matrix.png')

def run_project(root=None, dataset=None):
    start=time.time(); root=Path(root) if root else locate_root()
    results=root/'results';fig=results/'figures'
    for directory in [results,fig,results/'Confusion_Matrices',results/'Classification_Reports']:directory.mkdir(exist_ok=True)
    df=load_data(root, dataset); audit=audit_data(df)
    ids=np.arange(len(df))
    train,test=train_test_split(ids,test_size=.2,stratify=df.Is_Returned,random_state=SEED)
    assert not set(train)&set(test)
    X=df[FEATURES];y=df.Is_Returned
    Xtr,ytr=X.iloc[train].reset_index(drop=True),y.iloc[train].reset_index(drop=True)
    cv,folds,oof=cross_validate(Xtr,ytr)
    selected=str(cv.iloc[0].Model)
    thresholds=dict(zip(cv.Model,cv.OOF_Threshold))
    # Freeze selection BEFORE fitting/evaluating any holdout model.
    selection={'selected_model':selected,'selection_metric':'Mean 5-fold training CV average precision',
       'threshold':thresholds[selected],'threshold_objective':'Maximise returned-class F1 on training out-of-fold scores',
       'seed':SEED,'features':FEATURES,'holdout_used_for_selection':False}
    (results/'Selection_Before_Holdout.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    cv.to_csv(results/'CV_Model_Comparison.csv',index=False);folds.to_csv(results/'CV_Fold_Results.csv',index=False)
    pd.DataFrame({'Order_Record_ID':df.iloc[train].Order_Record_ID.to_numpy(),'Actual':ytr,
      **oof}).to_csv(results/'Training_OOF_Scores.csv',index=False)
    membership=pd.DataFrame({'Order_Record_ID':df.Order_Record_ID,'Split':np.where(np.isin(ids,test),'holdout','training')})
    membership.to_csv(results/'Split_Membership.csv',index=False)
    rows=[];fitted={};scores={}
    for name,model in models(ytr).items():
        print('Final fit:',name,flush=True);model.fit(Xtr,ytr)
        score=model.predict_proba(X.iloc[test])[:,1];scores[name]=score;fitted[name]=model
        for label,threshold in [('Default',.5),('OOF selected',thresholds[name])]:
            rows.append({'Model':name,'Operating_Point':label,'Threshold':threshold,'Selected_Model':name==selected,**metrics(y.iloc[test],score,threshold)})
        threshold=thresholds[name];pred=(score>=threshold).astype(int)
        safe=name.replace(' ','_')
        pd.DataFrame(confusion_matrix(y.iloc[test],pred),index=['Actual_success','Actual_return'],columns=['Pred_success','Pred_return']).to_csv(results/f'Confusion_Matrices/{safe}.csv')
        pd.DataFrame(classification_report(y.iloc[test],pred,output_dict=True,zero_division=0)).T.to_csv(results/f'Classification_Reports/{safe}.csv')
    holdout=pd.DataFrame(rows);holdout.to_csv(results/'Model_Comparison_Results.csv',index=False)
    selected_score=scores[selected];selected_threshold=thresholds[selected]
    pd.DataFrame({'Order_Record_ID':df.iloc[test].Order_Record_ID.to_numpy(),'Actual':y.iloc[test].to_numpy(),
      'Risk_Score':selected_score,'Review_Flag':(selected_score>=selected_threshold).astype(int)}).to_csv(results/'Selected_Model_Holdout_Scores.csv',index=False)
    # Training-only sensitivity: fixed selected family, positive COD population.
    positive=Xtr.COD.gt(0).to_numpy();Xp=Xtr.loc[positive].reset_index(drop=True);yp=ytr.loc[positive].reset_index(drop=True)
    sens_scores=np.zeros(len(yp));sens=[]
    for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(Xp,yp),1):
        m=models(yp.iloc[tr])[selected];m.fit(Xp.iloc[tr],yp.iloc[tr]);s=m.predict_proba(Xp.iloc[va])[:,1];sens_scores[va]=s
        sens.append({'Fold':fold,**metrics(yp.iloc[va],s,selected_threshold)})
    pd.DataFrame(sens).to_csv(results/'Positive_COD_Training_Sensitivity.csv',index=False)
    # Review capacity illustration is descriptive; it cannot retune the model.
    review=[];sort=np.argsort(-selected_score,kind='stable');yt=y.iloc[test].to_numpy()
    for fraction in [.05,.1,.2,.3]:
        k=int(np.ceil(len(test)*fraction));caught=int(yt[sort[:k]].sum())
        review.append({'Review_Fraction':fraction,'Orders_Reviewed':k,'Returns_Captured':caught,'Precision':caught/k,'Recall':caught/yt.sum(),'Lift_vs_Random':(caught/k)/yt.mean()})
    pd.DataFrame(review).to_csv(results/'Review_Capacity_Analysis.csv',index=False)
    metrics_selected=metrics(yt,selected_score,selected_threshold)
    summary={'audit':audit,'selection':selection,'train_rows':len(train),'test_rows':len(test),
      'train_returned':int(ytr.sum()),'test_returned':int(yt.sum()),'test_prevalence':float(yt.mean()),
      'selected_holdout':metrics_selected,'confusion_matrix':confusion_matrix(yt,selected_score>=selected_threshold).tolist(),
      'holdout_ci_95':bootstrap_ci(yt,selected_score,selected_threshold),
      'positive_cod_training_sensitivity':{'rows':len(yp),'returned':int(yp.sum()),'oof_ap':average_precision_score(yp,sens_scores),'mean_fold_ap':float(pd.DataFrame(sens).Average_Precision.mean())},
      'dataset_sha256':hashlib.sha256((Path(dataset).expanduser().resolve() if dataset else root/'data/Modalii_Lenshub_ML_Dataset_Final.csv').read_bytes()).hexdigest(),
      'versions':{p:importlib.metadata.version(p) for p in ['pandas','numpy','matplotlib','scikit-learn','imbalanced-learn','xgboost','scipy']},
      'python':platform.python_version(),'elapsed_seconds':time.time()-start,
      'validation_limit':'Retrospective random holdout; dataset was explored in the earlier draft. No dates or customer groups available. Not a prospective independent validation.'}
    (results/'Run_Summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    make_figures(df,cv,holdout,yt,selected_score,selected_threshold,fig)
    print(json.dumps(summary,indent=2),flush=True)
    return summary

if __name__=='__main__':run_project()
