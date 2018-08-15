
import json
prop = ['"asin"','"categories"','"price"','"salesRank"','"related"','"buy_after_viewing"','"imUrl"']

def check(str_):
    for sub_str in prop:
        if sub_str in str_:
            return True
    return False

def removeQuoteBetween(str_):
    s = str_[1]
    e = str_[-1]
    return s+str_[2:-1].replace('"','')+e

with open("json_test") as fs:
    for line in fs:
        line = line.replace('\\','')
        temp = line.strip().split(",")
        flag = 0
        for i in range(len(temp)):

            if '"description"' in temp[i]:
                temp_1 = temp[i].split(":")
                temp_1[1] = removeQuoteBetween(temp_1[1])
                temp[i] = temp_1[0]+': '+temp_1[1]
                flag = 1
                continue
            if '"title"' in temp[i]:
                #if temp[i-1][-1]!='"':
                #    temp[i-1] = temp[i-1]+'"'
                if len(temp[i-1])>0:
                    if not check(temp[i-1]):
                        if temp[i-1][-1]!='"':
                            temp[i-1]=temp[i-1]+'"'
                else:
                    if not check(temp[i-1]):
                        temp[i-1]+='"'
                temp_1 = temp[i].split(":")
                temp_1[1] = removeQuoteBetween(temp_1[1])
                temp[i] = temp_1[0]+': '+temp_1[1]
                flag = 1
                continue
            if not check(temp[i]) and flag==1:
                temp[i] = temp[i].replace('"','')
                continue
            if check(temp[i]):
                if flag==1:
                    if len(temp[i-1])>0:
                        if temp[i-1][-1]!='"':
                            temp[i-1]=temp[i-1]+'"'
                    else:
                        if not check(temp[i-1]):
                            temp[i-1]+='"'
                flag=0
                continue

        if flag==1:
            i = len(temp)
            temp[i-1] = temp[i-1].replace('}','')
            temp[i-1]+='"}'
        out_string = temp[0]
        for i in range(1,len(temp)):
            out_string+= ','+temp[i]

        print(out_string)
        data = json.loads(out_string)
        #print(data['description'])
        print(data['title'])
        #print(data['related']['also_viewed'])
