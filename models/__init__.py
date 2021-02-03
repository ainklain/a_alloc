from models import base_model_v3

model_list = {
    'default': base_model_v3.BaseModel,
    'cash_first': base_model_v3.CashFirstModel,
    'cash_first2': base_model_v3.CashFirstModel2,
    'epoch_adjust': base_model_v3.EpochAdjustModel,
    # 'model_v3': model_v3
}

