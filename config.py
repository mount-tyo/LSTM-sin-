# 学習データ数 : テストデータ数 = 8 : 2
# 学習データ数
N = 800   
# テストデータ数　
Nte = 200
# バッチサイズ 
bss = [40,20,10,5]
# bss = [40]
# 学習回数
n_epoch = 1000  
#n_epoch = 100
# ユニット数 [中間層1, 中間層2]
h_unitss = [[5,5],[9,9],[11,11],[20,20]]
#h_unitss = [[10, 10], [5, 5]]
# 活性化関数 (ReLU関数='relu', sigmoid関数='sig')
acts = ['sig', 'relu']

# 出力画像のファイルパス
# save_path = './Result/'
save_path = './LSTM/'
#save_path = './Test/'