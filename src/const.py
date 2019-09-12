import pandas as pd

train_identity = pd.read_csv("../data/train_identity.csv", )
train_transaction = pd.read_csv("../data/train_transaction.csv", )
test_identity = pd.read_csv("../data/test_identity.csv", )
test_transaction = pd.read_csv("../data/test_transaction.csv",)


BASE_COLUMNS  = list(train_transaction.columns) + list(train_identity.columns)

COLS_TO_DROP = ['V112', 'V315', 'V293', 'id_25', 'V135', 'V136', 'V284', 'V298', 'V300', 'V316',
                'V111', 'dist2', 'V105', 'V113', 'V104', 'id_24', 'id_22', 'V117', 'V121', 'V125',
                'V320', 'V103', 'V109', 'V118', 'V295', 'V303', 'V119', 'V134', 'V106', 'V281',
                'V120', 'V290', 'V98', 'V102', 'V115', 'V137', 'V123', 'id_08', 'V309', 'id_18',
                'V114', 'V321', 'V116', 'V133', 'V108', 'V301', 'V124', 'C3', 'V296', 'id_23',
                'V122', 'V129', 'id_26', 'V304', 'V110', 'V107', 'id_21', 'V286',  'id_27',
                'V297', 'V299', 'V311', 'V319', 'V305', 'V101', 'V289', 'id_07', 'V132', 'V318', 'D7']

CATEGORY_COLUMNS = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22',
                    'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33',
                    'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4',
                    'card6', 'M4','P_emaildomain', 'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1',
                    'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9', 'P_emaildomain_bin', 'DeviceInfo_c',
                    'id_30_c', 'P_emaildomain_suffix', 'R_emaildomain_bin', 'R_emaildomain_suffix']