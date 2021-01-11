---
title: "Auto-Scheduler"
date: 2020-04-15T15:34:30-04:00
categories:
  - Personal Project
tags:
  - Google API
  - Python
toc: true
toc_label:
toc_icon: 'bars'
classes: wide

---

# Overview
As a dog walker, my work schedule can vary quite a lot. I recieve an email of my schedule everyday from my boss, however I found it's a bit of a hassle to constantly pull up that email. In order to better keep track of all my clients' visits, I started to add events onto my Google Calendar widget I have on my phone's homepage. Although that helped a little, I still had to spend some time each day to create up to 15 events in my calendar. After some research I found that it's simple enough to use the Google API to automatically look at my inbox for that specific email, pick apart the text, and create the calendar events with the necessary information. 

<br />

|![image](/assets/images/schedule_1.png)|
|:--:|
|*Example of a schedule on my phone's calendar*|

<br />

First, download the following packages.

```python
import datetime
import pickle
import os.path
import base64
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

```

Next we must handle the authentication flow. This is taken from Google's own [quickstart guide](https://developers.google.com/gmail/api/quickstart/python). It's a good idea to enable the Gmail API for your Cloud Platform project at this point as well. Since we will also be using the Google Calendar API, you should also enable that for the same project as well. The two .json files you download from each API should be put into the working directory of this script.

```python
# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
SCOPES2 = ['https://www.googleapis.com/auth/calendar']

# The file token.pickle stores the user's access and refresh tokens, and is
# created automatically when the authorization flow completes for the first time.

creds = None
if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)

creds2 = None
if os.path.exists('token2.pickle'):
    with open('token2.pickle', 'rb') as token2:
        creds2 = pickle.load(token2)
    # If there are no (valid) credentials available, let the user log in.
if not creds2 or not creds2.valid:
    if creds2 and creds2.expired and creds2.refresh_token:
        creds2.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES2)
        creds2 = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('token2.pickle', 'wb') as token2:
        pickle.dump(creds2, token2)

```
Now that we have the credentials to access our account, we can start by retrieving the emails in our inbox.

```python
# Call the Google APIs
service = build('gmail', 'v1', credentials=creds)
service2 = build('calendar', 'v3', credentials=creds2)

# Retrieve a list of emails in primary inbox
results = service.users().messages().list(userId='me', labelIds=['INBOX']).execute()
messages = results.get('messages', 'me')

```
Just so we're familiar with the content of what we're looking for, here is an example email I may recieve.

<br />

|![image](/assets/images/email_1.png)|
|:--:|
|*Example email of my schedule*|

<br />

We can parse through our inbox to find the appropriate emails by using this for loop.
```python
# Cycle through emails. Find first email that contains walks
if not messages:
    print("No messages found.")
    exit(1)
else:
    for message in messages:
        messageheader = service.users().messages().get(userId="me", id=message['id'], format="full").execute()
        headers = messageheader["payload"]["headers"] # Headers are an attribute of messages which we typically refer to as the subject of the email
        petsit = "Your Daily Walking and Pet Sitting email. Please do not reply to this automated email."
```
Notice that the *petsit* variable is a string that contains the exact subject line as the one found in the example email from the image above. These automated emails always have this specific subject line, so we can easily identify which messages in the inbox we need to look at.
```python        
        subject = [i['value'] for i in headers if i["value"] == petsit]
        if subject:
            content = base64.urlsafe_b64decode(messageheader["payload"]["body"]["data"])
            content = str(content, 'utf-8')
            content = content.replace("<br>", " ")
            idx = content.find(",")
            idx2 = content.find(".")
            if content.find("upcoming services", idx, idx2) > -1:
                print("\n" + content)
                print("\nHave walks!\n")
                break
            if content.find("no services", idx, idx2) > -1:
                print("\n" + content)
                print("\nNo walks tomorrow!")
                exit(1)       
```
The *subject* variable is a boolean that returns true if the message header (also known as the email's subject) contains the exact string found in the *subject* variable. The *content* variable contains the body of the actual email. We have to convert the character encoding of the email to UTF-8 in order to actually read it in console and we replace the break character with the space character to more easily read the message. We can use the *content* variable to differentiate between an email that has my schedule for the next day and an email that notifies me that I don't have work the next day. An example of such an email is shown below.

<br />

|![image](/assets/images/email_2.png)|
|:--:|
|*Example email when I have no work the next day|

<br />

To differentiate between the two types of emails, we have to find out if the body of the email has the phrase "upcoming services" or "no services". The *find()* function returns the index of the first instance of the string you provide. If no instance of that string is found, it returns -1. You may define a start and end index to search through when using the *find()* function. So we look for the phrases after the first instance of a comma and before the first instance of a period in the email. 

For example, if I have work the next day, it looks for the phrase "upcoming services" in the string "Your Walker/Sitter Number is: MAZ Here are your upcoming services on January 11, 2019 for A FRIEND FOR FIDO LLC". When it does find it, we break out of the for loop.

If I don't have work the next day, it looks for the phrase "no services" in the string "You have no services for A FRIEND FOR FIDO LLC on January 29, 2019"

This loop only finds the most recent email sent to me about my schedule. So if the most recent email says I have no walks the next day, the script ends right there. Otherwise we continue to parse through the most recent email to see when my walks are schedule for the upcoming workday.

Now we need to gather the date of the workday so we know what day to create the work event in our calendar.
```python
# Get date
idx = content.find("<b>") + 3
idx2 = content.find("</b>")
date = content[idx:idx2]
year = int(date[-4:])
day = date.find(",")
day = int(date[day-2:day])
month = date.find(" ")
month = date[:month]
month = datetime.datetime.strptime(month, '%B')
month = int(month.month)
d = datetime.date(year, month, day)
```

Next we cut out unnecessary info from the body of the email to more easily read the contents in the console.

```python
# Cut out unnecessary info
idx = content.find("You are being sent this email because")
content = content[:idx]
```
Next we create a list of the notes for each walk that are given to me in the email. These are specific instructions that the client would like me to have for the job. These notes will be added to the description section of the events created in my calendar.
```python
note = content
notes = []
while note.find("Notes:") > -1:
    idx = note.find("Notes:") - 3
    note = note[idx:]
    if note.find(date) > -1:
        idx2 = note.find(date)
        content = content.replace(note[:idx2], " ")
    elif note.find("Have a great day") > -1:
        idx2 = note.find("Have a great day")
        content = content.replace(note[:idx2], " ")
    notes.append("Friend for Fido: " + note[13:idx2])
    note = note[13:]
```
Next we create a list for every walk I have for the day and separate list for the corresponding pay. The pay can be determined by the length of walk and how many dogs there are. You can see, for example, by looking at the string between "Service:" and "Pet Name:" there can be "TWO DOG 30 MINUTE WALK" or "ONE DOG 30 MINUTE WALK" etc. The first instance of these kind of strings are added to the *pays* array. After figuring out the type of job it is, we can look at the string after "Pet Name:" to find the name of the pet and append it to the *walks* array. The while loops goes through every instance of "Pet Name:" in the email and adds all the walks and pays the the corresponding lists.

For example, 
```python
# Add walks and pay to arrays
walks = []
pays = []
total = 0
pet = content
while pet.find("Pet Name:") > -1:
    idx = pet.find("Pet Name:")
    idx2 = pet.find("Service:") + 18
    pays.append(pet[idx2:idx - 3])
    pet = pet[idx + 14:]
    idx = pet.find("<b>")
    walks.append(pet[:idx])
```
Now that we have the date, name of pet/event, and notes, all that is left to create the events are the start and end times.
A similar tactic is used to find out the start and end times of every walk. We look for the first instance of the hyphen character and look at the string before and after to find the start and end time for that walk. It loops through every instance of a hyphen character until all walks are accounted for and all start and end times are gathered.

```python
# Find start and end times
startTimes = []
endTimes = []
eTime = content
while eTime.find(" - ") > -1:
    idx = eTime.find(" - ") + 3
    sTime = eTime[idx-11:]
    eTime = eTime[idx:]
    startTimes.append(sTime[:8])
    endTimes.append(eTime[:8])
```

We now have all the information we need to create the events for the calendar. We will create a number of events equal to the number of walks found in the *walks* array. 

The datetime library has a handy function that converts strings to the datetime format, using the function *strptime()*. We then combine the time and date into a single datetime object using the *combine()* function. Now we can create the event object by filling out the appropriate information such as the event name, start and end times, and description.
```python
# Create an event for each walk and upload to calendar
for i in range(len(walks)):

    #combine time and date parameters into a single datetime object
    st = datetime.datetime.strptime(startTimes[i], "%I:%M %p")
    start = datetime.datetime.combine(d, st.time())
    et = datetime.datetime.strptime(endTimes[i], "%I:%M %p")
    end = datetime.datetime.combine(d, et.time())

    event = {
        'summary': walks[i],
        'description': notes[i],
        'start': {
            'dateTime': start.isoformat("T"),
            'timeZone': 'America/Chicago',
        },
        'end': {
            'dateTime': end.isoformat("T"),
            'timeZone': 'America/Chicago',
        },
        'reminders': {
            'useDefault': True,
        },
    }
    
    service2.events().insert(calendarId='primary', body=event).execute()
    print(event)
    print("new walk created")
```
We then tally up the total pay from the *pays* array and create the total pay event for that work day.
```python
    # Tally up total pay
    if pays[i].find("ONE DOG 30") > -1:
        total += 16
    elif pays[i].find("TWO DOG 30") > -1:
        total += 20
    elif pays[i].find("ONE DOG 45") > -1:
        total += 24
    elif pays[i].find("TWO DOG 45") > -1:
        total += 28
    elif pays[i].find("TWO DOG 60") > -1:
        total += 36
        
    # Create total pay event
    if i == len(walks) - 1:
        pay_event = {
            'summary': "$" + str(total),
            'description': "Friend for Fido:",
            'start': {
                'date': str(d),
                'timeZone': 'America/Chicago',
            },
            'end': {
                'date': str(d),
                'timeZone': 'America/Chicago',
            },
            'colorId': "2",
            'reminders': {
                'useDefault': True,
            },
        }
        service2.events().insert(calendarId='primary', body=pay_event).execute()
        print(pay_event)
exit(1)
```
With a bit of parsing know-how and just a little research into Google's APIs we can create a simple yet powerful script that saves us a lot of time, keeps us organized and on top of our daily workflow. 
