thrs = 0.7

dfTR_eval['Y_LR_pred_thrs'] = (dfTR_eval["Y_LR_prob_pos"] >= thrs) * 1

TP, FN, FP, TN = confusion_matrix(y_true= dfTR_eval["Y"], 
                                  y_pred= dfTR_eval["Y_LR_pred_thrs"], 
                                  labels=[1, 0]).ravel()

fpr = FP/(TN + FP)
tpr = TP/(TP + FN)

fpr, tpr