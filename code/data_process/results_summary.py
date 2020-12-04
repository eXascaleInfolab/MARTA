from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score, roc_curve, accuracy_score
import pandas as pd

def classification_report_csv(report,evaluation_file):
    report_data = []
    lines = report.split('\n')
    lines = [t for t in lines if len(t) > 1]
    for line in lines[1:-3]:
        row = {}
        row_data = line.split('      ')
        row_data = [t for t in row_data if len(t) > 1]
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(evaluation_file,mode='a', index = False)
    return report_data



def save_results(evaluation_file,y_test,y_pred):
    print('accuracy %s' % accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    report = classification_report(y_test, y_pred)
    report_data = classification_report_csv(report, evaluation_file)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    auc_score = auc(recall, precision)
    print('PR AUC: %.3f' % auc_score)
    # calculate roc auc
    roc_auc = roc_auc_score(y_test, y_pred)
    print('ROC AUC %.3f' % roc_auc)
    # print("c=",c)
    with open(evaluation_file, 'a') as f:
        f.write('accuracy, %s\n' % accuracy_score(y_test, y_pred))
        f.write('auprc, %s\n' % auc_score)
        f.write('auc, %s\n' % roc_auc)
    return report_data