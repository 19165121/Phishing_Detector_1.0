import tkinter as tk
from tkinter import messagebox

# Function to check if the text contains phishing keywords
def is_phishing(text):
    phishing_keywords = ['password', 'account', 'verify', 'update', 'urgent', 'confirm', 'login', 'bank', 'security',
                         'information', 'verify', 'payment', 'credit', 'social', 'validate', 'immediately', 'unauthorized',
                         'suspend', 'restriction', 'alert', 'invalid', 'expire', 'fraud', 'limit', 'win', 'winner', 'claim',
                         'request', 'warning', 'suspicious', 'reset', 'verify', 'urgent', 'blocked', 'account', 'verify',
                         'update', 'identity', 'personal', 'confidential', 'login', 'SSN', 'PIN', 'secure', 'private',
                         'authentication', 'direct', 'fake', 'authenticate', 'verification', 'phishing', 'phisher',
                         'hacked', 'fraudulent', 'scam', 'login', 'username', 'password', 'login', 'password', 'click',
                         'link', 'email', 'email', 'address', 'email', 'phone', 'call', 'urgent', 'verify', 'log', 'site',
                         'secure', 'confirm', 'confidential', 'bank', 'account', 'credit', 'personal', 'security', 'fraud',
                         'reset', 'unauthorized', 'identity', 'suspicious', 'alert', 'warning', 'restricted', 'limit',
                         'expire', 'win', 'winner', 'claim', 'payment', 'request', 'suspend', 'update', 'validate',
                         'immediately', 'information', 'private', 'authenticate', 'notification', 'update', 'validate',
                         'verification', 'confirm', 'secure', 'log', 'authenticate', 'identity', 'invalid']
    
    keywords_count = sum(1 for keyword in phishing_keywords if keyword in text)

    if keywords_count >= 3:
        score = 50 + (keywords_count - 3) * 10.2
    else:
        score = keywords_count * 10.2

    return min(score, 100)

# Function to check for phishing
def check_phishing():
    user_text = text_entry.get("1.0", "end-1c")
    
    if len(user_text) > 2000:
        result = messagebox.askquestion("Text Too Large", "The entered text is too large (maximum 2000 characters).\nDo you want to truncate it?")
        if result == "yes":
            user_text = ' '.join(user_text.split())  # Remove multiple spaces
            user_text = user_text[:2000]
            text_entry.delete("1.0", tk.END)
            text_entry.insert(tk.END, user_text)
        else:
            return
    
    score = is_phishing(user_text)
    
    if score > 0:
        result_label.config(text=f"Phishing Probability: {score:.2f}%", fg="red")
    else:
        result_label.config(text="Phishing Probability: 0%", fg="green")

# Create the main window
root = tk.Tk()
root.title("Phishing Detector")

# Create widgets
label = tk.Label(root, text="Enter the text to analyze:")
label.pack(pady=10)

text_entry = tk.Text(root, height=5, width=50)
text_entry.pack(padx=10, pady=5)

check_button = tk.Button(root, text="Check", command=check_phishing)
check_button.pack(pady=10)

result_label = tk.Label(root, text="", fg="black")
result_label.pack()

# Start the GUI
root.mainloop()
