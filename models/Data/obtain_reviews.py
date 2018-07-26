# this script filters out reviews from the amazon data to test out the gender prediction models...

def get_reviews(fname_s,fname_d,cnt=10):
    with open(fname_d,"w") as ft:
        with open(fname_s) as fs:
            for line in fs:
                temp = line.strip().split('|')
                if len(temp)==8:
                    review = temp[5]
                    ft.write(review)
                    ft.write('\n')
                    cnt-=1
                    if cnt==0:
                        break


if __name__=="__main__":
    get_reviews("../../amazon_female.csv","sample_female")
    get_reviews("../../amazon_male.csv","sample_male")
