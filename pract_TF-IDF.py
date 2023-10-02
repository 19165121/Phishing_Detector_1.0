# Import necessary modules for the first code snippet
import tkinter as tk  # Import the tkinter library for creating a GUI
from tkinter import messagebox, StringVar  # Import specific components from tkinter
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TF-IDF vectorizer from scikit-learn
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression model from scikit-learn
import csv  # Import the CSV module for reading and writing CSV files
import string  # Import the string module for string manipulation and utilities
import pandas as pd  # Import the pandas library and alias it as 'pd' for data manipulation and analysis
import numpy as np  # Import the numpy library and alias it as 'np' for numerical computations
import spacy  # Import the spacy library for natural language processing tasks

#load spacy english mode
nlp = spacy.load("en_core_web_sm")





# Define a function named 'extensions' that takes a list of words as input
def extensions(words):
    # Initialise empty lists to store found emails and URLs
    emails_found = []
    urls_found = []

    # Define lists of email and URL suffixes for pattern matching
    email_suffixes = ['.us', '.uk', '.ca', '.au', '.de', '.fr', '.jp', '.it', '.es', '.nl', '.br', '.ru', '.cn', '.in', '.mx', '.za', '.ch', '.at', '.se', '.dk', '.no', '.fi', '.nz', '.sg', '.kr', '.ar', '.co.uk', '.jp', '.cn', '.br', '.es', '.ca', '.au', '.de', '.it', '.nl', '.ru', '.fr', '.in', '.mx', '.za', '.ch', '.at', '.se', '.dk', '.no', '.fi', '.nz', '.sg', '.kr', '.com', '.net', '.org', '.edu', '.gov', '.mil', '.int', '.biz', '.info', '.name', '.pro', '.museum', '.aero', '.coop', '.edu.au', '.edu.sg', '.eu', '.gov.au', '.gov.cn', '.gov.uk', '.gov.za', '.idv', '.mil.au', '.mil.cn', '.mil.uk', '.mil.za', '.museum.au', '.museum.sg', '.net.au', '.net.sg', '.org.au', '.org.sg', '.net', '.com', '.online', '.app', '.web', '.cc', '.im', '.ly', '.link']
    url_suffixes = ['.us', '.uk', '.ca', '.au', '.de', '.fr', '.jp', '.it', '.es', '.nl', '.br', '.ru', '.cn', '.in', '.mx', '.za', '.ch', '.at', '.se', '.dk', '.no', '.fi', '.nz', '.sg', '.kr', '.ar', '.co.uk', '.jp', '.cn', '.br', '.es', '.ca', '.au', '.de', '.it', '.nl', '.ru', '.fr', '.in', '.mx', '.za', '.ch', '.at', '.se', '.dk', '.no', '.fi', '.nz', '.sg', '.kr', '.com', '.net', '.org', '.edu', '.gov', '.mil', '.int', '.biz', '.info', '.name', '.pro', '.museum', '.aero', '.coop', '.edu.au', '.edu.sg', '.eu', '.gov.au', '.gov.cn', '.gov.uk', '.gov.za', '.idv', '.mil.au', '.mil.cn', '.mil.uk', '.mil.za', '.museum.au', '.museum.sg', '.net.au', '.net.sg', '.org.au', '.org.sg', '.net', '.com', '.online', '.app', '.web', '.cc', '.im', '.ly', '.link']

    # Iterate through each word in the input list
    for word in words:
        # Check if any email suffix is in the word and if it contains "@" character
        if any(suffix in word for suffix in email_suffixes) and "@" in word:
            emails_found.append(word)
        # Check if any URL suffix is in the word
        elif any(suffix in word for suffix in url_suffixes):
            urls_found.append(word)

    # Return the found emails and URLs
    return emails_found, urls_found





# Function to remove stop words using spaCy
def remove_stop_words(text):
    # Process the input text using spaCy
    doc = nlp(text)

    # Create a list of cleaned words by filtering out stop words
    cleaned_words = [token.text for token in doc if not token.is_stop]

    # Join the cleaned words back into a single string
    cleaned_text = ' '.join(cleaned_words)

    # Return the cleaned text
    return cleaned_text

# Function to lemmatise text using spaCy
def lemmatize_text(text):
    # Process the input text using spaCy
    doc = nlp(text)

    # Create a list of lemmatised words
    lemmatized_words = [token.lemma_ for token in doc]

    # Join the lemmatized words back into a single string
    lemmatized_text = ' '.join(lemmatized_words)

    # Return the lemmatised text
    return lemmatized_text





# Function to validate and process input text
def validate_text():
    # Get text from the input_box widget and strip leading/trailing spaces
    text = input_box.get("1.0", tk.END).strip()

    # Split the text into words
    words = text.split()

    # Find emails and URLs in the text
    emails_found, urls_found = extensions(words)

    # Check if there are more than one email or URL in the text
    if len(emails_found) > 1 or len(urls_found) > 1:
        # Show an error message if more than one email or URL is found
        messagebox.showerror("Error", "Only one email or URL per instance is allowed in the text. Please use the check button to fix it")
        return
    else:
        # Update the global 'confirmed_text' variable
        global confirmed_text
        text = input_box.get("1.0", tk.END).strip()

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

        # Remove extra spaces again and convert to lowercase
        cleaned_text = ' '.join(cleaned_text.split())
        cleaned_text = ' '.join(cleaned_other_words + emails_found + urls_found).lower()

        # Perform stop word removal and lemmatisation
        cleaned_text = remove_stop_words(cleaned_text)
        lemmatized_text = lemmatize_text(cleaned_text)

        # Clear the input_box widget and insert the processed text
        input_box.delete("1.0", tk.END)
        input_box.insert(tk.END, lemmatized_text)
        confirmed_text = lemmatized_text

        # Enable various buttons and disable the input_box
        analyze_button.config(state=tk.NORMAL)
        add_button.config(state=tk.NORMAL)
        check_button.config(state=tk.NORMAL)
        classify_website_label_button.config(state=tk.NORMAL)
        classify_address_label_button.config(state=tk.NORMAL)
        input_box.config(state=tk.DISABLED)

        # Update character count after processing
        update_character_count()





# Function to reset the input text and enable relevant buttons
def reset_text():
    # Enable the input_box widget and clear its content
    input_box.config(state=tk.NORMAL)
    input_box.delete("1.0", tk.END)     # Delete text

    # Reset the character count label and result label
    char_counter.config(text="Characters: 0/2000")
    result_label.config(text="", foreground="black")

    # Enable various buttons
    analyze_button.config(state=tk.NORMAL)
    add_button.config(state=tk.NORMAL)
    check_button.config(state=tk.NORMAL)
    classify_website_label_button.config(state=tk.NORMAL)
    classify_address_label_button.config(state=tk.NORMAL)

    # Hide and reset the label_menu widget
    label_menu.pack_forget()
    label_var.set("Select Label")

    # Enable the confirm_button
    confirm_button.config(state=tk.NORMAL)

    # Disable certain buttons
    analyze_button.config(state=tk.DISABLED)
    add_button.config(state=tk.DISABLED)
    classify_website_label_button.config(state=tk.DISABLED)
    classify_address_label_button.config(state=tk.DISABLED)





# Function to process the dataset
def process_dataset():
    # Load the dataset from a CSV file named "dataset.csv"
    dataset = pd.read_csv("dataset.csv")

    # Define a function to check if a word ends with any of the specified suffixes
    def has_suffix(word, suffixes):
        return any(word.endswith(suffix) for suffix in suffixes)

    # Define a function to remove URLs, email addresses, and words ending with a "." followed by letters
    def remove_links_emails_and_dot_words(text):
        if isinstance(text, str):
            # List of suffixes to identify URLs and email addresses
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

    # Apply the 'remove_links_emails_and_dot_words' function to the 'text' column of the dataset
    dataset["text"] = dataset["text"].apply(remove_links_emails_and_dot_words)

    # Save the edited dataset back to the "dataset.csv" file
    dataset.to_csv("dataset.csv", index=False)
    print("Dataset edited and saved.")

# Call the 'process_dataset' function to process the dataset
process_dataset()





# Function to clean empty rows from 'dataset.csv' file
def clean_empty_rows():
    # Read all lines from the file
    with open('dataset.csv', 'r') as file:
        lines = file.readlines()
    
    # Write non-empty lines back to the file
    with open('dataset.csv', 'w', newline='') as file:
        for line in lines:
            if line.strip():  # Check if the line is not empty
                file.write(line)

# Function to update character count in the user interface
def update_character_count(event=None):
    text = input_box.get("1.0", tk.END)
    char_count = len(text.strip())
    char_counter.config(text=f"Characters: {char_count}/2000")





# Function to classify text
def classify_text(text, vectorizer, model):
    text_vectorized = vectorizer.transform([text])
    probabilities = model.predict_proba(text_vectorized)[0]
    phishing_probability = probabilities[model.classes_.tolist().index("Phishing")] * 100
    return phishing_probability





# Function to analyse input text
def analyze_text():
    update_character_count()
    text = input_box.get("1.0", "end-1c").strip()
    
    if text:
        # Load the dataset
        dataset = pd.read_csv("dataset.csv")
        
        # Drop rows with missing text or label
        dataset = dataset.dropna(subset=['text', 'label'])
        
        # Split the dataset into features and labels
        X = dataset['text']
        y = dataset['label']
        
        # Initialize TfidfVectorizer and LogisticRegression
        vectorizer = TfidfVectorizer()
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





# Function to classify website label
def classify_website_label():
    # Update the character count in the user interface
    update_character_count()
    
    # Check if the entered text is too long (maximum 2000 characters)
    if len(input_box.get("1.0", "end-1c")) > 2000:
        # Show a warning message if the text is too long
        messagebox.showwarning("Text Too Long", "The entered text is too long (maximum 2000 characters).")
        return
    
    # Load the dataset
    dataset = pd.read_csv("dataset.csv")
    
    # Drop rows with missing website or label
    dataset = dataset.dropna(subset=['website', 'label'])
    
    # Extract websites and labels from the dataset
    websites = dataset['website'].tolist()
    labels = dataset['label'].tolist()
    
    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer()
    
    # Vectorize the website data
    X = vectorizer.fit_transform(websites)
    
    # Initialize LogisticRegression model
    model = LogisticRegression()
    
    # Fit the model
    model.fit(X, labels)
    
    # Get the text from the input box
    text = input_box.get("1.0", tk.END).strip()
    
    if text:
        # Process the text (convert to lowercase)
        processed_text = text.lower()
        
        # Classify the text using the trained model
        phishing_probability = classify_text(processed_text, vectorizer, model)
        
        # Update the result label based on the classification result
        if phishing_probability >= 50:
            result_label.config(text=f"Warning! Phishing attempt detected ({phishing_probability:.2f}%)", foreground="red")
        else:
            result_label.config(text=f"Legitimate ({phishing_probability:.2f}%)", foreground="green")
    else:
        result_label.config(text="Enter a website.")


        


# Function to classify address label
def classify_address_label():
    # Update the character count in the user interface
    update_character_count()
    
    # Check if the entered text is too long (maximum 2000 characters)
    if len(input_box.get("1.0", "end-1c")) > 2000:
        # Show a warning message if the text is too long
        messagebox.showwarning("Text Too Long", "The entered text is too long (maximum 2000 characters).")
        return
    
    # Load the dataset
    dataset = pd.read_csv("dataset.csv")
    
    # Drop rows with missing address or label
    dataset = dataset.dropna(subset=['address', 'label'])
    
    # Extract addresses and labels from the dataset
    addresses = dataset['address'].tolist()
    labels = dataset['label'].tolist()
    
    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer()
    
    # Vectorize the address data
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
        
        # Classify the text using the trained model
        phishing_probability = classify_text(processed_text, vectorizer, model)
        
        # Update the result label based on the classification result
        if phishing_probability >= 50:
            result_label.config(text=f"Warning! Phishing attempt detected ({phishing_probability:.2f}%)", foreground="red")
        else:
            result_label.config(text=f"Legitimate ({phishing_probability:.2f}%)", foreground="green")
    else:
        result_label.config(text="Enter an address.")





# Function to add data to the dataset
def add_to_dataset():
    # Get the text from the input box and strip leading/trailing spaces
    text = input_box.get("1.0", tk.END).strip()
    
    # Check if the text is missing
    if not text:
        # Show a warning message if the text is missing
        messagebox.showwarning("Missing Text", "Please enter the text before learning.")
        return
    
    # Call the 'show_label_menu' function to display the label menu
    show_label_menu()





# Function to show the label menu
def show_label_menu():
    # Display the label menu with padding
    label_menu.pack(padx=5, pady=5)
    
    # Display the confirm button with padding
    confirm_button.pack(padx=5, pady=5)
    
    # Disable the add_button
    add_button.config(state=tk.DISABLED)
    
    # Display the analyze_button on the left side with padding
    analyze_button.pack(side=tk.LEFT, padx=5)





# Function to confirm the label and detect emails
def confirm_label():
    # Get the selected label from label_var
    label = label_var.get()
    
    # Check if a label is selected and not the default "Select Label"
    if label and label != "Select Label":
        # Get the text from the input box and strip leading/trailing spaces
        text = input_box.get("1.0", tk.END).strip()
        
        # Check if the text is not empty and not the default "Enter some text"
        if text and text != "Enter some text":
            # Convert the text to lowercase and split it into words
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

            # Ask for confirmation and process the dataset file
            result = messagebox.askyesno("Question", "Do you want to confirm the text you entered?")
            if result:
                # Read and process the dataset file to clean URLs
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

            # Hide the label menu and reset UI elements
            label_menu.pack_forget()
            label_var.set("Select Label")
            add_button.config(state=tk.NORMAL)
            analyze_button.pack(side=tk.LEFT, padx=5)
            confirm_button.pack_forget()
            input_box.delete("1.0", tk.END)
            result_label.config(text="", foreground="black")
        else:
            # Show a warning if the text is missing
            messagebox.showwarning("Missing Text", "Please enter the text before confirming.")
    else:
        # Show a warning if no label is selected
        messagebox.showwarning("Select a Label", "Please select a label.")





# Function to show results of emails and URLs found
def show_email_and_url_results(emails_found, urls_found):
    # Create a new top-level window for displaying results
    result_window = tk.Toplevel(window)
    result_window.title("Flag the emails and/or URLs you want to keep (max 1 each)")
    result_window.geometry("700x300")
    
    # Create a frame for displaying email results
    email_frame = tk.Frame(result_window)
    email_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create a label for email results
    email_label = tk.Label(email_frame, text="Emails Found:", font=("Helvetica", 12, "bold"))
    email_label.pack(anchor="w")
    
    # Create a list to store email checkboxes and their associated variables
    email_checkboxes = []
    
    # Create a checkbox for each found email
    for email in emails_found:
        email_var = tk.StringVar()
        email_check = tk.Checkbutton(email_frame, text=email, variable=email_var)
        email_check.pack(anchor="w")
        email_checkboxes.append((email, email_var))
    
    # Create a frame for displaying URL results
    url_frame = tk.Frame(result_window)
    url_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create a label for URL results
    url_label = tk.Label(url_frame, text="URLs Found:", font=("Helvetica", 12, "bold"))
    url_label.pack(anchor="w")
    
    # Create a list to store URL checkboxes and their associated variables
    url_checkboxes = []
    
    # Create a checkbox for each found URL
    for url in urls_found:
        url_var = tk.StringVar()
        url_check = tk.Checkbutton(url_frame, text=url, variable=url_var)
        url_check.pack(anchor="w")
        url_checkboxes.append((url, url_var))





    # update the text based on the presence of emails or URLs
    def update_text():
        selected_emails = [email for email, email_var in email_checkboxes if email_var.get() == "1"]
        selected_urls = [url for url, url_var in url_checkboxes if url_var.get() == "1"]

        if len(selected_emails) <= 1 and len(selected_urls) <= 1:
            text = input_box.get("1.0", tk.END)
            for email in emails_found:
                if email not in selected_emails:
                    text = text.replace(email, "")
            for url in urls_found:
                if url not in selected_urls:
                    text = text.replace(url, "")
            input_box.delete("1.0", tk.END)
            input_box.insert(tk.END, text)
            result_window.destroy()
            input_box.edit_reset()
        else:
            messagebox.showwarning("Multiple Flags", "You can only select at most one flag per category.")

    update_button = tk.Button(result_window, text="Update Text", command=update_text)
    update_button.pack(pady=10)





# Function to check emails and URLs
def check_email_and_url():
    # Get the text from the input_box and remove leading/trailing whitespace
    text = input_box.get("1.0", tk.END).strip()
    
    if text:
        # Split the text into words
        words = text.split()
        
        # Call the 'extensions' function to find emails and URLs
        emails_found, urls_found = extensions(words)
        
        if emails_found or urls_found:
            # Display the results of found emails and URLs
            show_email_and_url_results(emails_found, urls_found)
        else:
            # If no emails or URLs found, display a message
            result_label.config(text="No emails or URLs found.", foreground="black")
    else:
        # If no text is entered, display a message
        result_label.config(text="Enter a message.", foreground="black")

# Processes two functions back to back
def rimo():
    confirm_label()  # Confirm the label entered by the user
    process_dataset()  # This will Process the whole dataset


    


# Create the main window
window = tk.Tk()
window.title("Phishing Detector 1.0")
window.geometry("800x1000")

# Create GUI elements
input_label = tk.Label(window, text="Enter some text")
input_label.pack(pady=10)

# Text input box
input_box = tk.Text(window, width=90, height=54)
input_box.pack(pady=5)

# Character counter label
char_counter = tk.Label(window, text="Characters: 0/2000")
char_counter.pack()

# Label frame for dropdown
label_frame = tk.Frame(window)
label_frame.pack(pady=10)

# Dropdown menu for selecting a label
label_var = StringVar()
label_var.set("Select Label")
label_menu = tk.OptionMenu(label_frame, label_var, "Select Label", "Legitimate", "Phishing")
label_menu.config(width=15)

# Frame for buttons
button_frame = tk.Frame(window)
button_frame.pack(pady=10)

# Analyze Text button
analyze_button = tk.Button(button_frame, text="Analyze Text", command=analyze_text)
analyze_button.pack(side=tk.LEFT, padx=5)

# Learn button
add_button = tk.Button(button_frame, text="Learn", command=add_to_dataset)
add_button.pack(side=tk.LEFT, padx=5)

# Confirm button (hidden by default)
confirm_button = tk.Button(button_frame, text="Confirm", command=rimo)
confirm_button.pack(side=tk.LEFT, padx=5)
confirm_button.pack_forget()

# Check button
check_button = tk.Button(button_frame, text="Check", command=check_email_and_url)
check_button.pack(side=tk.LEFT, padx=5)

# Result label
result_label = tk.Label(window, text="", foreground="black")
result_label.pack()

# Analyze website label button
classify_website_label_button = tk.Button(button_frame, text="Analyse 🌐", command=classify_website_label)
classify_website_label_button.pack(side=tk.LEFT, padx=5)

# Analyze address label button
classify_address_label_button = tk.Button(button_frame, text="Analyse 📩", command=classify_address_label)
classify_address_label_button.pack(side=tk.LEFT, padx=5)

# Validate button
validate_button = tk.Button(button_frame, text="Validate", command=validate_text)
validate_button.pack(side=tk.LEFT, padx=5)

# Reset button
reset_button = tk.Button(button_frame, text="Reset", command=reset_text)
reset_button.pack(side=tk.LEFT, padx=5)

# Disable buttons at the beginning
analyze_button.config(state=tk.DISABLED)
add_button.config(state=tk.DISABLED)

# Disable the 'classify_website_label_button' and 'classify_address_label_button' buttons
classify_website_label_button.config(state=tk.DISABLED)  # Disable the website label classification button
classify_address_label_button.config(state=tk.DISABLED)  # Disable the address label classification button





# Connect the character count update function to the input change even
input_box.bind("<KeyRelease>", update_character_count)





# Start the GUI event loop
window.mainloop()
