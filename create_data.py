import pandas as pd
import json
import copy
import requests
import shutil
import random

PERSUASION_STRATEGY = {0: 'Default', 1: 'Credibility', 2: 'Emotional', 3: 'Logical', 4: 'Personal', 5: 'Persona'}
SENTIMENT_STRATEGY = {0: 'Neutral', 1: 'Positive', -1: 'Negative'}

def preprocess_data(df):
    dataset = {}
    total_data = []
    train_data = []
    validation_data = []

    dialogue = []
    current_utterance = {}
    history = []

    dialogue_number = 1
    current_domain = "No Domain"
    belief_state = []

    cnt = 0
    img_cnt = 1
    file_path = './data/images/'

    for i in df.index:
        # print(dialogue_number)
        user_utterance = df["USER"][i]
        agent_utterance = df["AGENT"][i]
        if str(dialogue_number) == str(user_utterance):
            cnt += 1
            if dialogue_number != 1:
                total_data.append(dialogue)
            dialogue = []
            history = []
            belief_state = []
            current_domain = "No Domain"
            dialogue_number += 1
            continue

        # user_utterance = user_utterance.lower()
        # agent_utterance = agent_utterance.lower()

        if df["Persuasion Strategy"][i] == df["Persuasion Strategy"][i]:
            persuasion_strategy = int(df["Persuasion Strategy"][i])
        else:
            persuasion_strategy = int(0)

        if df["User Sentiment (-1 to 1)"][i] == df["User Sentiment (-1 to 1)"][i]:
            user_sentiment = int(df["User Sentiment (-1 to 1)"][i])
        else:
            user_sentiment = int(0)

        image_retrieval_status = False
        if df["MULTIMODAL_USER_UTTERANCE"][i] == df["MULTIMODAL_USER_UTTERANCE"][i]:
            img_url = df["MULTIMODAL_USER_UTTERANCE"][i]
            file_name = file_path + (str)(img_cnt) + '.jpg'
        else:
            img_url = "NA"
            file_name = "NA"

        img_url.strip()
        # print(img_url)

        if img_url != "NA":
            res = requests.get(img_url, stream=True, timeout=1000)

            if res.status_code == 200:
                image_retrieval_status = True
                res.raw.decode_content = True
                with open(file_name, 'wb') as f:
                    shutil.copyfileobj(res.raw, f)
                print('Image successfully Downloaded: ', file_name)
                img_cnt+=1
            else:
                print('Image Could not be retrieved')

        persuasion_strategy = PERSUASION_STRATEGY[persuasion_strategy]
        user_sentiment = SENTIMENT_STRATEGY[user_sentiment]

        if current_domain == "No Domain":
            tmp = df["Task Info"][i]
            if tmp == tmp:
                arr = tmp.split(',')
                if len(arr) >= 1:
                    current_domain = arr[0]

        if df["Slot-Value"][i] == df["Slot-Value"][i]:
            slot_tags = df["Slot-Value"][i]
        else:
            slot_tags = 'NA'

        slots = slot_tags.split(',')
        # print(slots)
        for slot in slots:
            arr = slot.split('-')
            if len(arr) >= 2:
                slot_name = arr[0].strip().lower()
                slot_val = arr[1].strip().lower()
                belief_state.append(current_domain + slot_name + " is " + slot_val)

        candidate = [agent_utterance]
        history.append(user_utterance)
        personality = [persuasion_strategy]
        sentiment = [user_sentiment]
        current_utterance['candidate'] = copy.deepcopy(candidate)
        current_utterance['history'] = copy.deepcopy(history)
        current_utterance['personality'] = copy.deepcopy(personality)
        current_utterance['sentiment'] = copy.deepcopy(sentiment)
        current_utterance['bstate'] = copy.deepcopy(belief_state)
        if image_retrieval_status:
            current_utterance['multimodal_utterance'] = copy.deepcopy(file_name)
        else:
            current_utterance['multimodal_utterance'] = "NA"

        dialogue.append(copy.deepcopy(current_utterance))
        history.append(agent_utterance)

    print(len(total_data))
    random.shuffle(total_data)
    itm_cnt = 1
    for itm in total_data:
        if itm_cnt <= 900:
            train_data.append(itm)
        else:
            validation_data.append(itm)
        itm_cnt += 1

    dataset["train"] = train_data
    dataset["validation"] = validation_data

    with open('data/btpData.json', 'w') as f:
        json.dump(dataset, f)

    with open('data/train_dials.json', 'w') as f:
        json.dump(train_data, f)

    with open('data/val_dials.json', 'w') as f:
        json.dump(validation_data, f)

    return

def main():
    df = pd.read_csv('./data/dataset.csv', encoding='windows-1252')

    preprocess_data(df)

if __name__ == "__main__":
    main()