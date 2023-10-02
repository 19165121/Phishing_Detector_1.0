# Import necessary libraries and modules
import tkinter as tk  # Import the tkinter library for creating GUI elements
from tkinter import messagebox, StringVar  # Import specific components from tkinter
from sklearn.feature_extraction.text import CountVectorizer  # Import CountVectorizer for text analysis
from sklearn.linear_model import LogisticRegression  # Import LogisticRegression for machine learning
import csv  # Import the CSV module for handling CSV files
import string  # Import the string module for string-related operations
import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations
import spacy  # Import the spacy library for natural language processing

# Load the English language model from spacy
nlp = spacy.load("en_core_web_sm")  # Load the English language model provided by spaCy





# Function to find email addresses and URLs in a list of words
def extensions(words):
    emails_found = []  # Initialise an empty list to store found email addresses
    urls_found = []    # Initialise an empty list to store found URLs

    # Lists of common email and URL suffixes
    email_suffixes = ['.us', '.uk', '.ca', '.au', '.de', '.fr', '.jp', '.it', '.es', '.nl', '.br', '.ru', '.cn', '.in', '.mx', '.za', '.ch', '.at', '.se', '.dk', '.no', '.fi', '.nz', '.sg', '.kr', '.ar', '.co.uk', '.jp', '.cn', '.br', '.es', '.ca', '.au', '.de', '.it', '.nl', '.ru', '.fr', '.in', '.mx', '.za', '.ch', '.at', '.se', '.dk', '.no', '.fi', '.nz', '.sg', '.kr', '.com', '.net', '.org', '.edu', '.gov', '.mil', '.int', '.biz', '.info', '.name', '.pro', '.museum', '.aero', '.coop', '.edu.au', '.edu.sg', '.eu', '.gov.au', '.gov.cn', '.gov.uk', '.gov.za', '.idv', '.mil.au', '.mil.cn', '.mil.uk', '.mil.za', '.museum.au', '.museum.sg', '.net.au', '.net.sg', '.org.au', '.org.sg']
    url_suffixes = ['.us', '.uk', '.ca', '.au', '.de', '.fr', '.jp', '.it', '.es', '.nl', '.br', '.ru', '.cn', '.in', '.mx', '.za', '.ch', '.at', '.se', '.dk', '.no', '.fi', '.nz', '.sg', '.kr', '.ar', '.co.uk', '.jp', '.cn', '.br', '.es', '.ca', '.au', '.de', '.it', '.nl', '.ru', '.fr', '.in', '.mx', '.za', '.ch', '.at', '.se', '.dk', '.no', '.fi', '.nz', '.sg', '.kr', '.com', '.net', '.org', '.edu', '.gov', '.mil', '.int', '.biz', '.info', '.name', '.pro', '.museum', '.aero', '.coop', '.edu.au', '.edu.sg', '.eu', '.gov.au', '.gov.cn', '.gov.uk', '.gov.za', '.idv', '.mil.au', '.mil.cn', '.mil.uk', '.mil.za', '.museum.au', '.museum.sg', '.net.au', '.net.sg', '.org.au', '.org.sg', '.com', '.online', '.app', '.web', '.cc', '.im', '.ly', '.link']

    for word in words:
        if any(suffix in word for suffix in email_suffixes) and "@" in word:
            # Check if the word contains an email suffix and the "@" symbol
            emails_found.append(word)  # If yes, add it to the emails_found list
        elif any(suffix in word for suffix in url_suffixes):
            # Check if the word contains a URL suffix
            urls_found.append(word)    # If yes, add it to the urls_found list

    return emails_found, urls_found  # Return the found email addresses and URLs





# Function to remove stop words using spaCy
def remove_stop_words(text):
    # Parse the input text using the spaCy language model
    doc = nlp(text)
    
    # Create a list of cleaned words by filtering out stop words
    cleaned_words = [token.text for token in doc if not token.is_stop]
    
    # Join the cleaned words back into a single string
    cleaned_text = ' '.join(cleaned_words)
    
    # Return the cleaned text without stop words
    return cleaned_text





# Function to lemmatize text using spaCy
def lemmatize_text(text):
    # Parse the input text using the spaCy language model
    doc = nlp(text)
    
    # Create a list of lemmatised words
    lemmatized_words = [token.lemma_ for token in doc]
    
    # Join the lemmatised words back into a single string
    lemmatized_text = ' '.join(lemmatized_words)
    
    # Return the lemmatised text
    return lemmatized_text





# Function to validate text input
def validate_text():
    # Get the text from the input_box and remove leading/trailing whitespace
    text = input_box.get("1.0", tk.END).strip()
    
    # Split the text into words
    words = text.split()
    
    # Use the 'extensions' function to find emails and URLs in the text
    emails_found, urls_found = extensions(words)
    
    # Check if there are more than one email or URL found
    if len(emails_found) > 1 or len(urls_found) > 1:
        # Show an error message if more than one email or URL is found
        messagebox.showerror("Error", "Only one email or URL per instance is allowed in the text. Please use the check button to fix it")
        return
    else:
        # Store the validated text in the global variable 'confirmed_text'
        global confirmed_text
        confirmed_text = text

        # Remove extra spaces
        text_without_extra_spaces = ' '.join(text.split())

        # Limit text to 2000 characters
        text_limited = text_without_extra_spaces[:2000]

        # Split the text into words
        words = text_limited.split()

        # Separate emails and URLs from other words
        emails_found, urls_found = extensions(words)
        other_words = [word for word in words if word not in emails_found and word not in urls_found]

        # Remove punctuation from non-email and non-URL words
        translator = str.maketrans("", "", string.punctuation)
        cleaned_other_words = [word.translate(translator) for word in other_words]

        # Combine cleaned words, emails, and URLs
        cleaned_text = ' '.join(cleaned_other_words + emails_found + urls_found)

        # Remove extra spaces again
        cleaned_text = ' '.join(cleaned_text.split())
        cleaned_text = ' '.join(cleaned_other_words + emails_found + urls_found).lower()

        # Perform stop word removal and lemmatization
        cleaned_text = remove_stop_words(cleaned_text)
        lemmatized_text = lemmatize_text(cleaned_text)

        # Clear the text in the input_box from the beginning to the end
        input_box.delete("1.0", tk.END)

        # Insert the lemmatized text at the end of the input_box
        input_box.insert(tk.END, lemmatized_text)

        # Update the 'confirmed_text' with the lemmatized text
        confirmed_text = lemmatized_text

        # Enable the following buttons and widgets
        analyze_button.config(state=tk.NORMAL)  # Enable the analyze button
        add_button.config(state=tk.NORMAL)      # Enable the add button
        check_button.config(state=tk.NORMAL)    # Enable the check button
        classify_website_label_button.config(state=tk.NORMAL)  # Enable website label classification button
        classify_address_label_button.config(state=tk.NORMAL)  # Enable address label classification button 

        # Disable the input_box to prevent further editing
        input_box.config(state=tk.DISABLED)

        # Update the character count after processing
        update_character_count()




        
# Function to reset the text and GUI elements
def reset_text():
    # Enable the input_box for text entry
    input_box.config(state=tk.NORMAL)
    
    # Clear the text in the input_box
    input_box.delete("1.0", tk.END)
    
    # Reset the character count label
    char_counter.config(text="Characters: 0/2000")
    
    # Reset the result label and set its text color to black
    result_label.config(text="", foreground="black")
    
    # Enable the following buttons and widgets
    analyze_button.config(state=tk.NORMAL)  # Enable the analyse button
    add_button.config(state=tk.NORMAL)      # Enable the add button
    check_button.config(state=tk.NORMAL)    # Enable the check button
    classify_website_label_button.config(state=tk.NORMAL)  # Enable website label classification button
    classify_address_label_button.config(state=tk.NORMAL)  # Enable address label classification button
    
    # Hide the label_menu (dropdown menu)
    label_menu.pack_forget()
    
    # Reset the label_var to "Select Label" in the dropdown menu
    label_var.set("Select Label")
    
    # Enable the confirm_button
    confirm_button.config(state=tk.NORMAL)
    
    # Disable the following buttons except for Reset and Confirm Text
    analyze_button.config(state=tk.DISABLED)  # Disable the analyse button
    add_button.config(state=tk.DISABLED)      # Disable the add button
    
    classify_website_label_button.config(state=tk.DISABLED)  # Disable website label classification button
    classify_address_label_button.config(state=tk.DISABLED)  # Disable address label classification button

    # Disable buttons except Reset and Confirm Text
    analyze_button.config(state=tk.DISABLED)
    add_button.config(state=tk.DISABLED)
    classify_website_label_button.config(state=tk.DISABLED)  # Disable website label classification button
    classify_address_label_button.config(state=tk.DISABLED)  # Disable address label classification button





# Function to process the dataset by removing URLs, email addresses, and words ending with specific suffixes
def process_dataset():
    # Load the dataset from a CSV file
    dataset = pd.read_csv("dataset.csv")

    # Function to check if a word ends with a specified list of suffixes
    def has_suffix(word, suffixes):
        return any(word.endswith(suffix) for suffix in suffixes)

    # Function to remove URLs, email addresses, and words ending with specific suffixes
    def remove_links_emails_and_dot_words(text):
        if isinstance(text, str):
            suffixes = [
                '.us', '.uk', '.ca', '.au', '.de', '.fr', '.jp', '.it', '.es', '.nl', '.br', '.ru', '.cn', '.in', '.mx', '.za',
                '.ch', '.at', '.se', '.dk', '.no', '.fi', '.nz', '.sg', '.kr', '.ar', '.co.uk', '.jp', '.cn', '.br', '.es',
                '.ca', '.au', '.de', '.it', '.nl', '.ru', '.fr', '.in', '.mx', '.za', '.ch', '.at', '.se', '.dk', '.no', '.fi',
                '.nz', '.sg', '.kr', '.com', '.net', '.org', '.edu', '.gov', '.mil', '.int', '.biz', '.info', '.name',
                '.pro', '.museum', '.aero', '.coop', '.edu.au', '.edu.sg', '.eu', '.gov.au', '.gov.cn', '.gov.uk', '.gov.za',
                '.idv', '.mil.au', '.mil.cn', '.mil.uk', '.mil.za', '.museum.au', '.museum.sg', '.net.au', '.net.sg',
                '.org.au', '.org.sg', '.net', '.com', '.online', '.app', '.web', '.cc', '.im', '.ly', '.link'
            ]
            words = text.split()
            cleaned_words = []
            emails_found = []
            urls_found = []

            for word in words:
                if "@" in word:
                    if has_suffix(word, suffixes):
                        emails_found.append(word)
                    else:
                        cleaned_words.append(word)
                elif has_suffix(word, suffixes):
                    urls_found.append(word)
                else:
                    cleaned_words.append(word)

            cleaned_text = ' '.join(cleaned_words)
            return cleaned_text

        return text
    
    # Apply the 'remove_links_emails_and_dot_words' function to clean the 'text' column in the dataset
    dataset["text"] = dataset["text"].apply(remove_links_emails_and_dot_words)
    
    # Save the edited dataset back to the CSV file
    dataset.to_csv("dataset.csv", index=False)
    
    print("Dataset edited and saved.")

# Call the 'process_dataset' function to clean the dataset
process_dataset()

# Function to clean empty rows in the dataset CSV file
def clean_empty_rows():
    with open('dataset.csv', 'r') as file:
        lines = file.readlines()
    
    with open('dataset.csv', 'w', newline='') as file:
        for line in lines:
            if line.strip():  # Check if the line is not empty
                file.write(line)





# Function to update the character count label
def update_character_count(event=None):
    # Update the character count label based on the text in the input_box.
    text = input_box.get("1.0", tk.END)
    char_count = len(text.strip())
    char_counter.config(text=f"Characters: {char_count}/2000")

# Function to classify text
def classify_text(text, vectorizer, model):
    # Classify text using a vectorizer and a machine learning model.
    text_vectorized = vectorizer.transform([text])
    probabilities = model.predict_proba(text_vectorized)[0]
    phishing_probability = probabilities[model.classes_.tolist().index("Phishing")] * 100
    return phishing_probability





# Function to analyze text and perform phishing classification
def analyze_text():
    # Update the character count label
    update_character_count()
    
    # Get the text from the input box
    text = input_box.get("1.0", "end-1c").strip()
    
    if text:
        # Load the dataset
        dataset = pd.read_csv("dataset.csv")
        
        # Drop rows with missing text or label
        dataset = dataset.dropna(subset=['text', 'label'])
        
        # Split the dataset into features and labels
        X = dataset['text']
        y = dataset['label']
        
        # Initialize CountVectorizer and LogisticRegression
        vectorizer = CountVectorizer()
        model = LogisticRegression()
        
        # Vectorise the text
        X_vectorized = vectorizer.fit_transform(X)
        
        # Fit the model
        model.fit(X_vectorized, y)
        
        # Perform classification and get probabilities
        text_vectorized = vectorizer.transform([text])
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Convert the predicted label to "Phishing" or "Legitimate"
        predicted_label = model.classes_[np.argmax(probabilities)]
        
        # Get the percentage probability for the predicted label
        probability_percentage = max(probabilities) * 100
        
        result_label.config(text=f"{predicted_label} ({probability_percentage:.2f}%)")
        
        # Adjust color based on prediction
        if predicted_label == "Phishing":
            result_label.config(foreground="red")
        else:
            result_label.config(foreground="green")
    else:
        result_label.config(text="Enter some text.", foreground="black")

    
    


# Function to classify the label of a website
def classify_website_label():
    # Update the character count label
    update_character_count()
    
    # Check if the entered text is too long (maximum 2000 characters)
    if len(input_box.get("1.0", "end-1c")) > 2000:
        messagebox.showwarning("Text Too Long", "The entered text is too long (maximum 2000 characters).")
        return
    
    # Load the dataset
    dataset = pd.read_csv("dataset.csv")
    
    # Drop rows with missing website or label
    dataset = dataset.dropna(subset=['website', 'label'])
    
    # Get lists of websites and labels from the dataset
    websites = dataset['website'].tolist()
    labels = dataset['label'].tolist()
    
    # Initialise CountVectorizer
    vectorizer = CountVectorizer()
    
    # Vectorise the website data
    X = vectorizer.fit_transform(websites)
    
    # Initialise LogisticRegression model
    model = LogisticRegression()
    
    # Fit the model
    model.fit(X, labels)
    
    # Get the text from the input box
    text = input_box.get("1.0", tk.END).strip()
    
    if text:
        # Process the text (convert to lowercase)
        processed_text = text.lower()
        
        # Classify the text using the 'classify_text' function
        phishing_probability = classify_text(processed_text, vectorizer, model)
        
        # Display the result based on the phishing probability
        if phishing_probability >= 50:
            result_label.config(text=f"Warning! Phishing attempt detected ({phishing_probability:.2f}%)", foreground="red")
        else:
            result_label.config(text=f"Legitimate ({phishing_probability:.2f}%)", foreground="green")
    else:
        result_label.config(text="Enter a website.")





# Function to classify the label of an address
def classify_address_label():
    # Update the character count label
    update_character_count()
    
    # Check if the entered text is too long (maximum 2000 characters)
    if len(input_box.get("1.0", "end-1c")) > 2000:
        messagebox.showwarning("Text Too Long", "The entered text is too long (maximum 2000 characters).")
        return
    
    # Load the dataset
    dataset = pd.read_csv("dataset.csv")
    
    # Drop rows with missing address or label
    dataset = dataset.dropna(subset=['address', 'label'])
    
    # Get lists of addresses and labels from the dataset
    addresses = dataset['address'].tolist()
    labels = dataset['label'].tolist()
    
    # Initialise CountVectorizer
    vectorizer = CountVectorizer()
    
    # Vectorise the address data
    X = vectorizer.fit_transform(addresses)
    
    # Initialize LogisticRegression model
    model = LogisticRegression()
    
    # Fit the model
    model.fit(X, labels)
    
    # Get the text from the input box
    text = input_box.get("1.0", tk.END).strip()
    
    if text:
        # Process the text (convert to lowercase)
        processed_text = text.lower()
        
        # Classify the text using the 'classify_text' function
        phishing_probability = classify_text(processed_text, vectorizer, model)
        
        # Display the result based on the phishing probability
        if phishing_probability >= 50:
            result_label.config(text=f"Warning! Phishing attempt detected ({phishing_probability:.2f}%)", foreground="red")
        else:
            result_label.config(text=f"Legitimate ({phishing_probability:.2f}%)", foreground="green")
    else:
        result_label.config(text="Enter an address.")





# Function to add data to the dataset
def add_to_dataset():
    # Get the text from the input box
    text = input_box.get("1.0", tk.END).strip()
    
    # Check if the entered text is empty
    if not text:
        messagebox.showwarning("Missing Text", "Please enter the text before learning.")
        return
    
    # Show the label menu
    show_label_menu()






# Function to show the label menu
def show_label_menu():
    # Show the label menu and confirm button, disable the add button
    label_menu.pack(padx=5, pady=5)
    confirm_button.pack(padx=5, pady=5)
    add_button.config(state=tk.DISABLED)
    analyze_button.pack(side=tk.LEFT, padx=5)





# Function to confirm the label and detect emails
def confirm_label():
    # Get the selected label from the label_var
    label = label_var.get()
    
    # Check if a label is selected and not the default "Select Label"
    if label and label != "Select Label":
        # Get the text from the input box
        text = input_box.get("1.0", tk.END).strip()
        
        # Check if the text is not empty and not the default "Enter some text"
        if text and text != "Enter some text":
            # Convert text to lowercase and split into words
            text = text.lower()
            words = text.split()

            # Detect emails and URLs in the text
            emails_found, urls_found = extensions(words)

            # Remove punctuation from words that are not emails or URLs
            translator = str.maketrans("", "", string.punctuation)
            text_without_punct = ' '.join([word if (word in emails_found or word in urls_found) else word.translate(translator) for word in words])

            # Check if the dataset file is empty
            with open('dataset.csv', 'r') as file:
                is_empty = file.read().strip() == ''

            # Append the data to the dataset.csv file
            with open('dataset.csv', 'a', newline='') as file:
                fieldnames = ['text', 'address', 'website', 'label']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if not is_empty:
                    file.write('\n')
                writer.writerow({'text': text_without_punct, 'address': ', '.join(emails_found), 'website': ', '.join(urls_found), 'label': label})

            # Ask for confirmation and process dataset file
            result = messagebox.askyesno("Question", "Do you want to confirm the text you entered?")
            if result:
                with open('dataset.csv', 'r') as file:
                    lines = file.readlines()
                with open('dataset.csv', 'w', newline='') as file:
                    for line in lines:
                        parts = line.strip().split(',')
                        if len(parts) >= 3:  
                            parts[2] = parts[2].replace("http://", "").replace("https://", "").replace("www.", "").split('/')[0]
                            file.write(','.join(parts) + '\n')
                        else:
                            file.write(line)

                # Clean empty rows from the dataset
                clean_empty_rows()

            # Reset GUI elements
            label_menu.pack_forget()
            label_var.set("Select Label")
            add_button.config(state=tk.NORMAL)
            analyze_button.pack(side=tk.LEFT, padx=5)
            confirm_button.pack_forget()
            input_box.delete("1.0", tk.END)
            result_label.config(text="", foreground="black")
        else:
            messagebox.showwarning("Missing Text", "Please enter the text before confirming.")
    else:
        messagebox.showwarning("Select a Label", "Please select a label.")





# Function to show results of emails and URLs found
def show_email_and_url_results(emails_found, urls_found):
    # Create a new window for displaying results
    result_window = tk.Toplevel(window)
    result_window.title("Flag the emails and/or URLs you want to keep (max 1 each)")
    result_window.geometry("700x300")
    
    # Frame for displaying emails
    email_frame = tk.Frame(result_window)
    email_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    email_label = tk.Label(email_frame, text="Emails Found:", font=("Helvetica", 12, "bold"))
    email_label.pack(anchor="w")
    
    email_checkboxes = []  # List to store email checkboxes
    
    # Create checkboxes for each email found
    for email in emails_found:
        email_var = tk.StringVar()
        email_check = tk.Checkbutton(email_frame, text=email, variable=email_var)
        email_check.pack(anchor="w")
        email_checkboxes.append((email, email_var))
    
    # Frame for displaying URLs
    url_frame = tk.Frame(result_window)
    url_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    url_label = tk.Label(url_frame, text="URLs Found:", font=("Helvetica", 12, "bold"))
    url_label.pack(anchor="w")
    
    url_checkboxes = []  # List to store URL checkboxes
    
    # Create checkboxes for each URL found
    for url in urls_found:
        url_var = tk.StringVar()
        url_check = tk.Checkbutton(url_frame, text=url, variable=url_var)
        url_check.pack(anchor="w")
        url_checkboxes.append((url, url_var))





    # Function to update the text based on selected emails and URLs
    def update_text():
        # Get selected emails and URLs
        selected_emails = [email for email, email_var in email_checkboxes if email_var.get() == "1"]
        selected_urls = [url for url, url_var in url_checkboxes if url_var.get() == "1"]

        # Check if only one flag per category is selected
        if len(selected_emails) <= 1 and len(selected_urls) <= 1:
            text = input_box.get("1.0", tk.END)

            # Remove emails and URLs that are not selected
            for email in emails_found:
                if email not in selected_emails:
                    text = text.replace(email, "")
            for url in urls_found:
                if url not in selected_urls:
                    text = text.replace(url, "")

            # Update the input box with the modified text        
            input_box.delete("1.0", tk.END)
            input_box.insert(tk.END, text)

            # Destroy the result window and reset text edits
            result_window.destroy()
            input_box.edit_reset()
        else:
            # Show a warning message if multiple flags are selected
            messagebox.showwarning("Multiple Flags", "You can only select at most one flag per category.")

    # Create a button to trigger the text update
    update_button = tk.Button(result_window, text="Update Text", command=update_text)
    update_button.pack(pady=10)


    


# Function to check emails and URLs in the input text
def check_email_and_url():
    # Get the text from the input box and strip leading/trailing spaces
    text = input_box.get("1.0", tk.END).strip()
    
    if text:
        # Split the text into words
        words = text.split()
        
        # Find emails and URLs in the text
        emails_found, urls_found = extensions(words)
        
        if emails_found or urls_found:
            # Show the results of found emails and URLs
            show_email_and_url_results(emails_found, urls_found)
        else:
            # Display a message when no emails or URLs are found
            result_label.config(text="No emails or URLs found.", foreground="black")
    else:
        # Display a message when no input message is provided
        result_label.config(text="Enter a message.", foreground="black")





# Function to confirm the label and process the dataset
def rimo():
    confirm_label()  # Call the confirm_label function to confirm the label and update the dataset
    process_dataset()  # Call the process_dataset function to process the dataset


    


# Create the main window
window = tk.Tk()
window.title("Phishing Detector 1.0")
window.geometry("800x1000")

# Create a label for input instructions
input_label = tk.Label(window, text="Enter some text")
input_label.pack(pady=10)

# Create a text input box for user text input
input_box = tk.Text(window, width=90, height=54)
input_box.pack(pady=5)

# Create a label for character count
char_counter = tk.Label(window, text="Characters: 0/2000")
char_counter.pack()

# Create a frame for label-related elements
label_frame = tk.Frame(window)
label_frame.pack(pady=10)

# Initialize a variable to hold the selected label
label_var = StringVar()
label_var.set("Select Label")

# Create a dropdown menu for label selection
label_menu = tk.OptionMenu(label_frame, label_var, "Select Label", "Legitimate", "Phishing")
label_menu.config(width=15)

# Create a frame for buttons
button_frame = tk.Frame(window)
button_frame.pack(pady=10)

# Create a button to analyze the entered text
analyze_button = tk.Button(button_frame, text="Analyze Text", command=analyze_text)
analyze_button.pack(side=tk.LEFT, padx=5)

# Create a button to learn and add data to the dataset
add_button = tk.Button(button_frame, text="Learn", command=add_to_dataset)
add_button.pack(side=tk.LEFT, padx=5)

# Create a button to confirm label and dataset entry
confirm_button = tk.Button(button_frame, text="Confirm", command=rimo)
confirm_button.pack(side=tk.LEFT, padx=5)
confirm_button.pack_forget()  # Initially, hide the Confirm button

# Create a button to check for emails and URLs in the text
check_button = tk.Button(button_frame, text="Check", command=check_email_and_url)
check_button.pack(side=tk.LEFT, padx=5)

# Create a label to display analysis results
result_label = tk.Label(window, text="", foreground="black")
result_label.pack()

# Create a button to analyze website-related data
classify_website_label_button = tk.Button(button_frame, text="Analyse 🌐", command=classify_website_label)
classify_website_label_button.pack(side=tk.LEFT, padx=5)

# Create a button to analyze address-related data
classify_address_label_button = tk.Button(button_frame, text="Analyse 📩", command=classify_address_label)
classify_address_label_button.pack(side=tk.LEFT, padx=5)

# Create a button to validate text entry
validate_button = tk.Button(button_frame, text="Validate", command=validate_text)
validate_button.pack(side=tk.LEFT, padx=5)

# Create a button to reset input and labels
reset_button = tk.Button(button_frame, text="Reset", command=reset_text)
reset_button.pack(side=tk.LEFT, padx=5)

# Disable buttons at the beginning
analyze_button.config(state=tk.DISABLED)
add_button.config(state=tk.DISABLED)
classify_website_label_button.config(state=tk.DISABLED)
classify_address_label_button.config(state=tk.DISABLED)



# Connect the character count update function to the input change even
input_box.bind("<KeyRelease>", update_character_count)


# Start the GUI event loop
window.mainloop()
