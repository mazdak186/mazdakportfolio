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

```python
import datetime
import pickle
import os.path
import base64
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


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

# Call the Google APIs
service = build('gmail', 'v1', credentials=creds)
service2 = build('calendar', 'v3', credentials=creds2)

# Retrieve a list of emails in primary inbox
results = service.users().messages().list(userId='me', labelIds=['INBOX']).execute()
messages = results.get('messages', 'me')

# Cycle through emails. Find first email that contains walks
if not messages:
    print("No messages found.")
    exit(1)
else:
    for message in messages:
        messageheader = service.users().messages().get(userId="me", id=message['id'], format="full").execute()
        headers = messageheader["payload"]["headers"]
        # This subject is different from "Your dog walking schedule..."
        petsit = "Your Daily Walking and Pet Sitting email. Please do not reply to this automated email."
        subject = [i['value'] for i in headers if i["value"] == petsit]
        #print(messageheader['snippet'])
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

# Cut out unnecessary info
idx = content.find("You are being sent this email because")
content = content[:idx]
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

# Create a timedelta objects for CST and tomorrow
cstTimeDelta = datetime.timedelta(hours=-5)
tomorrow = datetime.timedelta(hours=32)

# Create a timezone instance for CST time zone
tzObject = datetime.timezone(cstTimeDelta, name="CST")

# Replace the time zone with CST
dateTime = datetime.datetime.today()
cstTimeNow = dateTime.replace(tzinfo=tzObject)
cstTomorrow = cstTimeNow + tomorrow

# Get list of events tomorrow
future_events = service2.events().list(calendarId='primary',
                                       timeMin=cstTimeNow.isoformat("T", "seconds"),
                                       timeMax=cstTomorrow.isoformat("T", "seconds")).execute()

# Create an event for each walk and upload to calendar
for i in range(len(walks)):
    duplicate = False

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
