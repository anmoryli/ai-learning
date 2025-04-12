import random

epochs = 100
accuracy = 0.5
loss = 0.9

with open("training_log,txt",'w',encoding='utf8') as f:
    f.write("epoch\taccuracy\tloss\n")
    for epoch in range(1,epochs+1):
        accuracy += random.uniform(0,0.005)
        loss -= random.uniform(0,0.005)
        accuracy = min(1, accuracy)
        loss = max(0,loss)
        f.write(f'{epoch}\t{accuracy:.3f}\t{loss:.3f}\n')
        print(f'Epoch:{epoch}\tAccuracy:{accuracy}\tLoss:{loss}')