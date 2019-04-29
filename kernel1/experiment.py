from dependencies import *



with open(result_save_path + '_train.pickle', 'rb') as f:
    data = pickle.load(f)
    truevals_tra = data['label']
    predvals_tra = data['prediction']
f.close()

with open(result_save_path + '_dev.pickle', 'rb') as f:
    data = pickle.load(f)
    truevals_dev = data['label']
    predvals_dev = data['prediction']
f.close()

with open(result_save_path + '_test.pickle', 'rb') as f:
    data = pickle.load(f)
    truevals_test = data['label']
    predvals_test = data['prediction']
f.close()

with open(gender_dump_path, 'rb') as f:
    gender = pickle.load(f)
    train_gender = gender['train']
    test_gender = gender['test']
    dev_gender = gender['dev']


male_index_tra = np.where(np.array(train_gender)==0)
female_index_tra = np.where(np.array(train_gender) == 1)
male_index_dev = np.where(np.array(dev_gender) == 0)
female_index_dev = np.where(np.array(dev_gender) == 1)
male_index_test = np.where(np.array(test_gender) == 0)
female_index_test = np.where(np.array(test_gender) == 1)


def calculate_f1_score(truevals,predvals):
    f1_macro = f1_score(truevals, predvals, average='macro')
    f1_micro = f1_score(truevals, predvals, average='micro')
    f1_weighted = f1_score(truevals, predvals, average='weighted')
    f1_none = f1_score(truevals, predvals, average=None)
    return f1_macro,f1_micro,f1_weighted,f1_none


def print_result(f1_macro_O,f1_micro_O,f1_weighted_O,f1_none_O,
                 f1_macro_M, f1_micro_M, f1_weighted_M, f1_none_M,
                 f1_macro_F, f1_micro_F, f1_weighted_F, f1_none_F,
                 acc_O,acc_M,acc_F):

    print(" ".center(35, ' ') + "O".center(35, ' ') + "M".center(35, ' ') + "F".center(35, ' ') + "B".center(35, ' '))
    print("F1-score-macro:".center(35, ' ') + str(f1_macro_O).center(35, ' ') + str(f1_macro_M).center(35,' ') + str(f1_macro_F).center(35, ' ') + str((f1_macro_F / f1_macro_M)).center(35, ' '))
    print("F1-score_micro:".center(35, ' ') + str(f1_micro_O).center(35, ' ') + str(f1_micro_M).center(35,' ') + str(f1_micro_F).center(35, ' ') + str((f1_micro_F / f1_micro_M)).center(35, ' '))
    print("F1-score_weighted:".center(35, ' ') + str(f1_weighted_O).center(35, ' ') + str(f1_weighted_M).center(35, ' ') + str(f1_weighted_F).center(35, ' ') + str((f1_weighted_F / f1_weighted_M)).center(35,' '))
    print("F1-score_none:".center(35, ' ') + str(f1_none_O).center(35, ' ') + str(f1_none_M).center(35, ' ') + str(f1_none_F).center(35, ' ') + str((f1_none_F / f1_none_M)).center(35, ' '))
    print("Accuracy:".center(35, ' ') + str(acc_O).center(35, ' ') + str(acc_M).center(35, ' ') + str(acc_F).center(35, ' ') + str((acc_F / acc_M)).center(35, ' '))



# Train result
f1_macro_tra_O,f1_micro_tra_O,f1_weighted_tra_O,f1_none_tra_O = calculate_f1_score(truevals_tra,predvals_tra)
f1_macro_tra_M,f1_micro_tra_M,f1_weighted_tra_M,f1_none_tra_M = calculate_f1_score(truevals_tra[male_index_tra],predvals_tra[male_index_tra])
f1_macro_tra_F,f1_micro_tra_F,f1_weighted_tra_F,f1_none_tra_F = calculate_f1_score(truevals_tra[female_index_tra],predvals_tra[female_index_tra])

acc_tra_O = accuracy_score(truevals_tra,predvals_tra)
acc_tra_M = accuracy_score(truevals_tra[male_index_tra],predvals_tra[male_index_tra])
acc_tra_F = accuracy_score(truevals_tra[female_index_tra],predvals_tra[female_index_tra])

print("Train result:")
print_result(f1_macro_tra_O,f1_micro_tra_O,f1_weighted_tra_O,f1_none_tra_O,
             f1_macro_tra_M, f1_micro_tra_M, f1_weighted_tra_M, f1_none_tra_M,
             f1_macro_tra_F, f1_micro_tra_F, f1_weighted_tra_F, f1_none_tra_F,
             acc_tra_O, acc_tra_M, acc_tra_F)
print('\n')



# Dev result
f1_macro_dev_O,f1_micro_dev_O,f1_weighted_dev_O,f1_none_dev_O = calculate_f1_score(truevals_dev,predvals_dev)
f1_macro_dev_M,f1_micro_dev_M,f1_weighted_dev_M,f1_none_dev_M = calculate_f1_score(truevals_dev[male_index_dev],predvals_dev[male_index_dev])
f1_macro_dev_F,f1_micro_dev_F,f1_weighted_dev_F,f1_none_dev_F = calculate_f1_score(truevals_dev[female_index_dev],predvals_dev[female_index_dev])


acc_dev_O = accuracy_score(truevals_dev,predvals_dev)
acc_dev_M = accuracy_score(truevals_dev[male_index_dev],predvals_dev[male_index_dev])
acc_dev_F = accuracy_score(truevals_dev[female_index_dev],predvals_dev[female_index_dev])


print("Validation result:")
print_result(f1_macro_dev_O,f1_micro_dev_O,f1_weighted_dev_O,f1_none_dev_O,
             f1_macro_dev_M, f1_micro_dev_M, f1_weighted_dev_M, f1_none_dev_M,
             f1_macro_dev_F, f1_micro_dev_F, f1_weighted_dev_F, f1_none_dev_F,
             acc_dev_O, acc_dev_M, acc_dev_F)

print('\n')



# Test result
f1_macro_test_O,f1_micro_test_O,f1_weighted_test_O,f1_none_test_O = calculate_f1_score(truevals_test,predvals_test)
f1_macro_test_M,f1_micro_test_M,f1_weighted_test_M,f1_none_test_M = calculate_f1_score(truevals_test[male_index_test],predvals_test[male_index_test])
f1_macro_test_F,f1_micro_test_F,f1_weighted_test_F,f1_none_test_F = calculate_f1_score(truevals_test[female_index_test],predvals_test[female_index_test])


acc_test_O = accuracy_score(truevals_test,predvals_test)
acc_test_M = accuracy_score(truevals_test[male_index_test],predvals_test[male_index_test])
acc_test_F = accuracy_score(truevals_test[female_index_test],predvals_test[female_index_test])


print("Test result:")

print_result(f1_macro_test_O,f1_micro_test_O,f1_weighted_test_O,f1_none_test_O,
             f1_macro_test_M, f1_micro_test_M, f1_weighted_test_M, f1_none_test_M,
             f1_macro_test_F, f1_micro_test_F, f1_weighted_test_F, f1_none_test_F,
             acc_test_O,acc_test_M,acc_test_F)

print('\n')

