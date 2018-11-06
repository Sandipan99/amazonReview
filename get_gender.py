import gender_guesser.detector as gender
import json
import gzip
import string

translator = str.maketrans('', '', string.punctuation)

d = gender.Detector()

ft_m = open("amazon_male.csv","w")
c_m = 0
ft_m_m = open("amazon_mostly_male.csv","w")
c_m_m = 0
ft_f = open("amazon_female.csv","w")
c_f = 0
ft_m_f = open("amazon_mostly_female.csv","w")
c_m_f = 0
ft_n_d = open("amazon_undecided.csv","w")
c_n_d = 0



def Gender(name):
    name = name.translate(translator).split()[0]
    name = ''.join([i for i in name if not i.isdigit()])
    return d.get_gender(name,'usa')



count_r = 0
count_in = 0

with open("../amazon_review_with_gender","w") as ft:
    with gzip.open("../user_dedup.json.gz","rb") as fs:
        for line in fs:
            data = json.loads(line.strip())
            count_r+=1
            try:
                name = data['reviewerName']
                gender = Gender(name)
                review = data['reviewText']
                review = review.replace("|",",")
                help = data['helpful']
                h = str(help[0])+","+str(help[1])

                if gender=='male':
                    ft_m.write('%s|%s|%s|%s|%s|%s|%s|%s\n' % (data['reviewerID'],data['asin'],name,gender,h,review,data['overall'],data['unixReviewTime']))
                    c_m+=1
                elif gender=='mostly_male':
                    ft_m_m.write('%s|%s|%s|%s|%s|%s|%s|%s\n' % (data['reviewerID'],data['asin'],name,gender,h,review,data['overall'],data['unixReviewTime']))
                    c_m_m+=1
                elif gender=='female':
                    ft_f.write('%s|%s|%s|%s|%s|%s|%s|%s\n' % (data['reviewerID'],data['asin'],name,gender,h,review,data['overall'],data['unixReviewTime']))
                    c_f+=1
                elif gender=='mostly_female':
                    ft_m_f.write('%s|%s|%s|%s|%s|%s|%s|%s\n' % (data['reviewerID'],data['asin'],name,gender,h,review,data['overall'],data['unixReviewTime']))
                    c_m_f+=1
                else:
                    ft_n_d.write('%s|%s|%s|%s|%s|%s|%s|%s\n' % (data['reviewerID'],data['asin'],name,gender,h,review,data['overall'],data['unixReviewTime']))
                    c_n_d+=1

            except:
                count_in+=1
                print(c_m,c_m_m,c_f,c_m_f,c_n_d)
                continue

print (count_r)
print (count_in)


ft_m.close()
ft_m_m.close()
ft_f.close()
ft_m_f.close()
ft_n_d.close()

print(c_m,c_m_m,c_f,c_m_f,c_n_d)
