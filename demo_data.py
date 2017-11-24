import random
import csv

os=["Linux","Ubunut","Mac OS X","Windows 7","Windows 8","Android"]
browser = ["Chrome","Firefox","IE","Safari","Chrome Mobile"]
page_visits = 0
time_spent = 0
device_type = ["web","mobile"]
unique_visits = 0
Outcome = ["Contact Us","Request Demo","Buy from Us"]
print("[os,browser,device_type, page_visits, unique_visits, time_Spent]")
def get_rando():
	return list([random.randrange(len(os)),random.randrange(len(browser)),
	random.randrange(len(device_type)),random.randrange(30),
	random.randrange(10),random.randrange(1800),random.randrange(3)])

train_data=[]
test_data=[]
for i in range(200):
	train_data.append(get_rando())


myfile = open('track_train.csv','w')
myfile.write("200,6,os,Contact Us,Request Demo,Buy from Us\n")
wr = csv.writer(myfile, delimiter=',')
wr.writerows(train_data)
myfile.close()

for i in range(50):
	test_data.append(get_rando())

myfile = open('track_test.csv','w')
myfile.write("50,6,os,Contact Us,Request Demo,Buy from Us\n")
wr = csv.writer(myfile, delimiter=',')
wr.writerows(test_data)
myfile.close()
