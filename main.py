# from app import preview_linker
from data import *
# from app import *
from meta import *
from net import *
import streamlit as st
st.set_page_config(layout="wide")

@st.cache()
def init_data():
    DIRECTORY = 'data'
    data_dict = {'politifact': 'data/truth-detectiondeception-detectionlie-detection/politifact.csv', 'politifact_clean': 'data/truth-detectiondeception-detectionlie-detection/politifact_clean.csv', 'politifact_clean_binarized': 'data/truth-detectiondeception-detectionlie-detection/politifact_clean_binarized.csv'}
    clean_truth_data = PreprocessingDataset(data_dict['politifact_clean_binarized'], DIRECTORY, 'statement', 'veracity', ['source', 'link'])
    print('Data Loading Complete')

    return clean_truth_data

clean_truth_data = init_data()

BATCH_SIZE = 64
primary_data = clean_truth_data #secondary option of truth_data
train_len = int(len(primary_data)*0.8)
test_len = len(primary_data) - train_len

train_set, test_set = torch.utils.data.random_split(primary_data, [train_len, test_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# print(len(train_set))
# print(len(test_set))

num_feats = np.array([train_set[i][0].numpy() for i in range(len(train_set))])
num_labels = np.array([train_set[i][1] for i in range(len(train_set))])

a = iter(train_loader)
b = next(a)
b = np.asarray(b[0])
# print(b.shape)
inp_size = (b.shape)[1]


import itertools
ab = list(itertools.chain(*[i[0] for i in clean_truth_data]))
# print(len(ab))
ab = set([int(i) for i in ab])
emb_dim = len(ab)


print('Preprocessing Complete')


max_len = len(train_set[1][0])
ref_check = 1

@st.cache()
def model_load():
    feedforward = FeedForward(ref_check, inp_size).to(device)
    recurrent = RecurrentClassifier(emb_dim, inp_size, 50, ref_check, 2, dropout=0.2).to(device)

    return feedforward, recurrent

feedforward, recurrent = model_load()

def train(net, train_loader, LR, DECAY, EPOCHS):
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=DECAY)
    loss_func = torch.nn.BCEWithLogitsLoss()

    epochs = EPOCHS
    losses = []

    for step in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inp, labels = data
            if net == recurrent:
                inp, labels = inp.long().to(device), labels.float().to(device)
                optimizer.zero_grad()
                outputs = net(inp)
                cost = loss_func(torch.squeeze(outputs), torch.squeeze(labels))
            elif net == feedforward:
                inp, labels = inp.float().to(device), labels.float().to(device)
                optimizer.zero_grad()
                outputs = net(inp)
                cost = loss_func(torch.squeeze(outputs), labels)
            cost.backward()
            optimizer.step()

            running_loss += cost.item()
        print(f'Epoch: {step}   Training Loss: {running_loss/len(train_loader)}')
    print('Training Complete')  

    return losses

def eval(net, test_loader):
    total = 0
    acc = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=DECAY)

    for i, data in enumerate(test_loader):
        inp, labels = data
        optimizer.zero_grad()
        output = net(inp.float())
        output = output.detach().numpy()
        output = list(output)
        output = [list(i).index(max(i)) for i in output]
        
        for idx, item in enumerate(torch.tensor(output)):
            total += 1
            if item == labels[idx]:
                acc += 1
    print(f'{acc/total*100}%')


# In[116]:


# def model_load(net, PATH, name, export=True):
#     if export:
#         torch.save(net.state_dict(), PATH+name+'.pth')
#         return PATH+name+'.pth'
#     else:
#         net.torch.load_state_dict(torch.load(PATH + name + '.pth'))
#         return net

print('Model and Train/Eval Initialization Complete')


# # train(feedforward, train_loader, 1e-3, 5e-3, 200)
# model_load(feedforward, 'model_parameters/', 'linear_politifact')

# # train(recurrent, train_loader, 1e-3, 5e-3, 200)
# model_load(recurrent, 'model_parameters/', 'lstm_politifact')

token_basis = clean_truth_data.token


def tokenize_sequence(text_inp, tokenizer):
    text_inp = text_inp.lower().split('\n')
    tokenizer.fit_on_texts(text_inp)
    sequences = tokenizer.texts_to_sequences(text_inp)
    sequences = [i if i!=[] else [0] for i in tokenizer.texts_to_sequences(text_inp)]
    sequences = [i[0] for i in sequences]
    pad_len =  [0]*int(inp_size - len(sequences))
    sequences += pad_len
    return torch.FloatTensor(sequences)[:600]

# inp = tokenize_sequence(summ, token_basis)
# inp = inp[None, :]
# print(inp.shape)

def prediction(inp, model):
    output = model(inp)
    return output

@st.cache
def model_load(net, PATH, name, export=True):
    if export:
        torch.save(net.state_dict(), PATH+name+'.pth')
        return PATH+name+'.pth'
    else:
        net.torch.load_state_dict(torch.load(PATH + name + '.pth'))
        net.eval()

st.title("Welcome to omniscius veritatis.")
st.subheader('\tTo seek the truth means to seek your own bias.')

st.text("""
Built by @Dev Patel, this project aims to contribute to universal bias 
prediction and identification from both real readers and deep learning 
algorithms to help readers better inform themselves of their own 
cognitive biases and empower a fair, just, and truthful dialect.
""")

status = st.radio("Select an Input Type: ", ("Link to Article", "Raw Text Input"))



if (status == "Link to Article"):
    preview = st.text_input("Paste the source link", "Enter here")
    if(st.button('Submit')):
        authart, publ, timg, allimg, tit, summ = meta_extract(preview)  
        sent = sentiment(summ)

        inp = tokenize_sequence(summ, token_basis)
        inp = inp[:600]
        inp = inp[None, :]
        # print(inp.shape)

        feedforward_template = FeedForward(ref_check, inp_size).to(device)
        recurrent_template = RecurrentClassifier(emb_dim, inp_size, 50, ref_check, 2, dropout=0.2).to(device)    
        model_load(feedforward_template, 'model_parameters/', 'linear_politifact')
        model_load(recurrent_template, 'model_parameters/', 'lstm_politifact')

        print("Feedforward and LSTM Loading Pretrained Loading Complete")

        # feedforward_template.eval()
        # recurrent_template.eval()

        output_linear = '0 ERROR'
        output_lstm = '1 ERROR' #check for error without passing error

        output_linear = F.sigmoid(prediction(inp, feedforward_template)).round()
        output_lstm = F.sigmoid(prediction(inp.long(), recurrent_template))

        all_types = list(pd.read_csv(data_dict['politifact_clean'])['veracity'].unique())


        st.info(f"Link: {preview}")
        st.info(f"Author: {authart}")
        st.info(f"Date: {publ}")

        st.info(f"Top Image: {timg}")
        st.info(f"All Images: {allimg}")
        st.info(f"Summary of {tit}")
        st.info(f" \t{summ}")

        st.success(f"{sent}")


        if output_linear == 0:
            output_linear = f"Little Bias: Prediction = {output_linear}"
            st.success(output_linear)
        elif output_linear == 1:
            output_linear = f"Substantial Bias: Prediction = {output_linear}"
            st.error(output_linear)

        statement_type = ''
        final_output = f"Veracity -> {statement_type}: {output_lstm}"
        if output_lstm <= 0.25:
            statement_type = 'True'
            st.success(final_output)
        elif 0.25 < output_lstm <= 0.5:
            statement_type = 'Mostly True'
            st.info(final_output)
        elif 0.5 < output_lstm <= 0.75:
            statement_type = 'Mostly False'
            st.warning(final_output)
        elif 0.75 < output_lstm <= 1:
            statement_type = 'False'
            st.error(final_output)
        elif output_lstm > 1:
            statement_type = 'Pants on Fire!'
            st.info(final_output)

else:
    inp_raw = st.text_inpt("Paste the raw data")
    if(st.button('Submit')):
        inp_raw = format_raw_text(inp_raw)
        preview = inp_raw.replace('uxd', ' ')     
        summ = preview  
        empty_msg = 'None'
        authart = empty_msg
        publ = empty_msg
        timg = empty_msg
        allimg = empty_msg
        tit = empty_msg

        sent = sentiment(summ)

        inp = tokenize_sequence(summ, token_basis)
        inp = inp[:600]
        inp = inp[None, :]
        # print(inp.shape)

        feedforward_template = FeedForward(ref_check, inp_size).to(device)
        recurrent_template = RecurrentClassifier(emb_dim, inp_size, 50, ref_check, 2, dropout=0.2).to(device)    
        model_load(feedforward_template, 'model_parameters/', 'linear_politifact')
        model_load(recurrent_template, 'model_parameters/', 'lstm_politifact')

        print("Feedforward and LSTM Loading Pretrained Loading Complete")

        # feedforward_template.eval()
        # recurrent_template.eval()

        output_linear = '0 ERROR'
        output_lstm = '1 ERROR' #check for error without passing error

        output_linear = F.sigmoid(prediction(inp, feedforward_template)).round()
        output_lstm = F.sigmoid(prediction(inp.long(), recurrent_template))

        all_types = list(pd.read_csv(data_dict['politifact_clean'])['veracity'].unique())


        st.info(f"Link: {preview}")
        st.info(f"Author: {authart}")
        st.info(f"Date: {publ}")

        st.info(f"Top Image: {timg}")
        st.info(f"All Images: {allimg}")
        st.info(f"Summary of {tit}")
        st.info(f" \t{summ}")

        if list(sent.values())[:-1].index(max(list(sent.values()[:-1]))) == 0:
            st.error(f"{sent}")
        elif list(sent.values())[:-1].index(max(list(sent.values()[:-1]))) == 1:
            st.success(f"{sent}")
        else:
            st.warning(f"{sent}")


        if output_linear == 0:
            output_linear = f"Little Bias: Prediction = {output_linear}"
            st.success(output_linear)
        elif output_linear == 1:
            output_linear = f"Substantial Bias: Prediction = {output_linear}"
            st.error(output_linear)

        statement_type = ''
        final_output = f"Veracity -> {statement_type}: {output_lstm}"
        if output_lstm <= 0.25:
            statement_type = 'True'
            st.success(final_output)
        elif 0.25 < output_lstm <= 0.5:
            statement_type = 'Mostly True'
            st.info(final_output)
        elif 0.5 < output_lstm <= 0.75:
            statement_type = 'Mostly False'
            st.warning(final_output)
        elif 0.75 < output_lstm <= 1:
            statement_type = 'False'
            st.error(final_output)
        elif output_lstm > 1:
            statement_type = 'Pants on Fire!'
            st.info(final_output)


# %%
