# performs pre-processing on the names to improve accuracy of gender guesser....

import gender_guesser.detector as gender
import string

translator = str.maketrans('', '', string.punctuation)

d = gender.Detector(case_sensitive=False)

total_count = 0
male = 0
female = 0


with open('../amazon_undecided.csv') as fs:
    for line in fs:
        temp = line.strip().split('|')
        if len(temp)==8:
            total_count+=1
            if total_count%1000000==0:
                print('accessed review: ',total_count)
            try:
                name = temp[2].translate(translator).split()[0].lower()
                name = ''.join([i for i in name if not i.isdigit()])
                gender = d.get_gender(name,'usa')
                if gender=='male':
                    male+=1
                elif gender=='female':
                    female+=1
                else:
                    print(name)
            except:
                print('Except: ',name)
            #print(name,gender)


print('total count:',total_count)
print('male: ',male)
print('female: ',female)
