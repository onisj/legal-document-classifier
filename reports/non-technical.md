### Non-Technical Report

I’m excited to share my work on the Legal Document Classifier project for the AI Engineer role at LawPavilion! This project is all about helping lawyers and legal professionals quickly figure out what area of law a case report belongs to, like civil procedure or criminal law, using a smart computer system I built.

#### **What I Did**
I started with a set of 200 legal case documents and cleaned them up by removing messy parts like extra codes or symbols, making them easier for the system to understand. I then created a way to label each document based on key phrases, grouping them into categories. To make the system smart, I tried different methods: some simple ones that work like a calculator with text, and a more advanced one called BERT that learns from lots of examples. I also added a feature to balance the data so all categories get a fair chance. Finally, I built a website and an app where users can type in a case report, choose a method, and get an answer right away.

#### **How It Works**
I made two tools: a backend API that does the heavy lifting of classifying the text, and a friendly chat-like app where users can interact with it. The app shows the predicted category (e.g., "Civil Procedure") and how confident the system is, along with some details. I saved the best versions of my tools so they can be used later.

#### **What I Found**
The simpler method I used worked really well, correctly guessing the category about 70% of the time, which I’m proud of! The advanced BERT method, however, struggled a bit, only getting it right about 50% of the time. This is because I didn’t have enough documents to train it properly, but I tried my best to improve it with some clever tricks.

#### **Challenges I Faced**
Training the advanced system was tough because I only had a small amount of data and was using my personal laptop, which isn’t super powerful. I adjusted things like how much data it looks at and how long it trains, but it still needs more examples to get better. I also learned that specialized legal data might help a lot.

#### **What’s Next**
I’d love to collect more case reports to make the system even smarter, maybe using a model designed for legal texts. I also want to make the app easier to use on different devices by adding something called Docker. Plus, I’d like to track how people use it to see what works best.

I hope this project shows my skills and enthusiasm for solving real-world problems at LawPavilion!