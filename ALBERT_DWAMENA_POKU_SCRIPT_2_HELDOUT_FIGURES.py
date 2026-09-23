"""SCRIPT 2 — held-out tables/ROC/PR/confusion matrices. CONSUMES Script 1; REFITS NOTHING."""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from ALBERT_DWAMENA_POKU_COMMON import *
ROOT=root_from_script(__file__); D=dirs(ROOT)
metrics=pd.read_csv(D['results']/'script1_heldout_metrics.csv')
# Figure 4.2: metric comparison
long=metrics.melt(id_vars='model',value_vars=['accuracy','precision','recall','f1','roc_auc','auc_pr'],var_name='metric',value_name='score')
piv=long.pivot(index='model',columns='metric',values='score')
ax=piv.plot(kind='bar',figsize=(11,6)); ax.set_ylim(0,1.05); ax.set_ylabel('Score'); ax.set_title('Figure 4.2: Held-out Model Performance'); plt.xticks(rotation=20,ha='right'); plt.tight_layout(); plt.savefig(D['figures']/'Figure_4_2_model_performance.png',dpi=300); plt.close()
# ROC + PR from canonical predictions only
fig,ax=plt.subplots(figsize=(8,6))
fig2,ax2=plt.subplots(figsize=(8,6))
for name in metrics['model']:
    p=pd.read_csv(D['artifacts']/f'script1_predictions_{name}.csv')
    fpr,tpr,_=roc_curve(p.y_true,p.y_prob); ax.plot(fpr,tpr,label=name.replace('_',' ').title())
    pr,re,_=precision_recall_curve(p.y_true,p.y_prob); ax2.plot(re,pr,label=name.replace('_',' ').title())
    cm=confusion_matrix(p.y_true,p.y_pred)
    disp=ConfusionMatrixDisplay(cm); disp.plot(values_format='d'); plt.title(f'Confusion Matrix: {name.replace("_"," ").title()}'); plt.tight_layout(); plt.savefig(D['figures']/f'confusion_matrix_{name}.png',dpi=300); plt.close()
ax.plot([0,1],[0,1],'--'); ax.set(xlabel='False Positive Rate',ylabel='True Positive Rate',title='Figure 4.3: ROC Curves'); ax.legend(); fig.tight_layout(); fig.savefig(D['figures']/'Figure_4_3_ROC_curves.png',dpi=300); plt.close(fig)
ax2.set(xlabel='Recall',ylabel='Precision',title='Precision–Recall Curves'); ax2.legend(); fig2.tight_layout(); fig2.savefig(D['figures']/'precision_recall_curves.png',dpi=300); plt.close(fig2)
print('SCRIPT 2 COMPLETE — no refit performed.')
