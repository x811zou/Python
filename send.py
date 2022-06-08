import smtplib

from email.mime.text import MIMEText
from email.utils import formataddr
from sys import argv
 
sender = 'zx0319@163.com'
receivers = ['xz195@duke.edu']
 
message = MIMEText('You job has been successfully executed, please check it on you cluster')
message['From'] = formataddr(["shane","zx0319@163.com"])
message['To'] =  formataddr(["test","xz195@duke.edu"])
 
message['Subject'] = "Job Finished " + argv[1]


smtp = smtplib.SMTP_SSL('smtp.163.com', 994)  

smtp.login("zx0319@163.com", "19910319Aa")
smtp.sendmail(sender, receivers, message.as_string()) 

smtp.quit()
