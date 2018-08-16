
import json
import re
prop = ['"asin"','"categories"','"price"','"salesRank"','"related"','"buy_after_viewing"','"imUrl"']
#prop = ['"asin"','"price"','"salesRank"','"related"','"buy_after_viewing"','"imUrl"']

def check(str_):
    for sub_str in prop:
        if sub_str in str_:
            return True
    return False

def removeQuoteBetween(str_):
    s = str_[1]
    e = str_[-1]
    return s+str_[2:-1].replace('"','')+e

cnt = 0

with open("metadata.json",'rb') as fs:
    for line in fs:
        cnt+=1
        line = line.decode()
        line = line.replace('\\','')
        temp = line.strip().split(",")
        flag = 0
        for i in range(len(temp)):
            print(temp[i],i)

            if '"description"' in temp[i]:
                print ("inside case 1")
                temp_1 = temp[i].split(":")
                temp_1[1] = removeQuoteBetween(temp_1[1])
                temp[i] = temp_1[0]+': '+temp_1[1]
                flag = 1
                continue
            if '"title"' in temp[i]:
                #if temp[i-1][-1]!='"':
                #    temp[i-1] = temp[i-1]+'"'
                print("inside case 2")
                if len(temp[i-1])>0:
                    if not check(temp[i-1]):
                        if (temp[i-1][-1]!='"')and(temp[i-1][-1]!='}')and(temp[i-1][-1]!=']'):
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
                temp[i-1] = temp[i-1][:-2]
                #print("inside case 3")
                temp[i] = temp[i].replace('"','')
                temp[i] = temp[i].replace('[','(')
                temp[i] = temp[i].replace(']',')')
                #print(temp[i])
                #print('-----------------------')
                continue
            if check(temp[i]):
                print("inside case 4")
                if flag==1:
                    if len(temp[i-1])>0:
                        print("inside case 4.1")
                        if (temp[i-1][-1]!='"')and(temp[i-1][-1]!='}')and(temp[i-1][-1]!=']'):
                            temp[i-1]=temp[i-1]+'"'
                    else:
                        print("inside case 4.2")
                        temp[i-1]+='"'
                        '''
                        if len(temp[i-1])>0:
                            if not check(temp[i-1]):
                                #temp[i-1]+='"'
                                if (temp[i-1][-1]!='"')and(temp[i-1][-1]!='}')and(temp[i-1][-1]!=']'):
                                    temp[i-1]=temp[i-1]+'"'
                        '''
                flag=0
                continue

            '''
            else:
                print("inside case 5")
                if len(temp[i])>0:
                    x = [k for k,v in enumerate(temp[i]) if v=='"']
                    print("len x: ",len(x))
                    if len(x)==0:
                        continue
                    if len(x)%2==0:
                        continue
                    else:
                        if x[0]==1:
                            y = [k for k,v in enumerate(temp[i]) if re.match('\w',v)]
                            temp[i] = temp[i][:1]+temp[i][1:y[-1]+1]+'"'+temp[i][y[-1]+1:]
                            continue
            '''



        if flag==1:
            i = len(temp)
            temp[i-1] = temp[i-1].replace('}','')
            temp[i-1]+='"}'
        out_string = temp[0]
        for i in range(1,len(temp)):
            if len(temp[i])>0:
                out_string+= ','+temp[i]

        #check for contents within square brackets
                

        try:
            data = json.loads(out_string)
        except:
            print(out_string)
            print(cnt)
            print(line)
            break
        #print(data['description'])
        #print(data['title'])
        #print(data['related']['also_viewed'])
