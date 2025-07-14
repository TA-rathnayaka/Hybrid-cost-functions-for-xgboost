import numpy as np

###########################################
#   standard EEL distibution based link   #
###########################################
def eel(x,beta=0.75,lamda=0.75):
    p=np.power(1-np.power((1+np.exp(x)),-lamda),beta)
    return p

###########################################
#      Dice Score                         #
###########################################
def DSC(y_true, y_pred):
    # Calculate true positives (tp), false negatives (fn) and false positives (fp)
    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * (1-y_pred))
    fp = np.sum((1-y_true) * y_pred)
    # Calculate Dice score
    dice_class = (2*tp)/((2*tp) + fp + fn)
    return dice_class  

###########################################
#      MCC                                #
###########################################
def MCC(y_true, y_pred):
    # Calculate true positives (tp),true negative(tn), false negatives (fn) and false positives (fp)
    tp = np.sum(y_true * y_pred)
    tn=  np.sum((1-y_true) * (1-y_pred))
    fn = np.sum(y_true * (1-y_pred))
    fp = np.sum((1-y_true) * y_pred)
    # Calculate Dice score
    num = (tp*tn)-(fp*fn)
    den =(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return np.divide(num,np.power(den,0.5))


