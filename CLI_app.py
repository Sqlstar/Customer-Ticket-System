# ===== IMPORTS =====
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# ===== INITIALIZATION =====
# Data files
TICKETS_FILE = 'tickets.csv'
AUDIT_FILE = 'audit_log.csv'
KNOWLEDGE_FILE = 'knowledge_base.json'

# Initialize data structures
def init_data():
    data = {}
    
    # Tickets
    if os.path.exists(TICKETS_FILE):
        data['tickets'] = pd.read_csv(TICKETS_FILE, parse_dates=['created_at', 'resolved_at'])
    else:
        data['tickets'] = pd.DataFrame(columns=[
            'ticket_id', 'subject', 'description', 'priority', 
            'assigned_team', 'status', 'created_at', 'resolved_at', 'resolution'
        ]).astype({
            'created_at': 'datetime64[ns]',
            'resolved_at': 'datetime64[ns]'
        })
    
    # Audit log
    if os.path.exists(AUDIT_FILE):
        data['audit_log'] = pd.read_csv(AUDIT_FILE, parse_dates=['timestamp'])
    else:
        data['audit_log'] = pd.DataFrame(columns=[
            'timestamp', 'ticket_id', 'action', 'user', 'details'
        ]).astype({'timestamp': 'datetime64[ns]'})
    
    # Knowledge base
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE) as f:
            data['knowledge_base'] = json.load(f)
    else:
        data['knowledge_base'] = {
            'login issues': "Try password reset at example.com/reset",
            'payment problems': "Contact billing at billing@example.com"
        }
    
    return data

# ===== CORE CLASS =====
class TicketSystem:
    def __init__(self):
        self.data = init_data()
        self.team_capacity = {
            'Technical Support': 5,
            'Customer Service': 8,
            'Billing': 4,
            'General Inquiry': 6
        }
        self.team_load = {team: 0 for team in self.team_capacity}
        self.users = {
            'admin': {
                'password': 'admin123',
                'role': 'admin',
                'team': 'Admin'
            },
            'support1': {
                'password': 'sup123',
                'role': 'support',
                'team': 'Technical Support'
            },
            'agent1': {
                'password': 'agent123',
                'role': 'agent',
                'team': 'Customer Service'
            }
        }
        self.current_user = None
        self.init_models()
    
    def init_models(self):
        self.text_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000,
                ngram_range=(1,2),
                stop_words='english'
            ))
        ])
        self.clf = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            max_depth=10,
            random_state=42
        )
        self.train_models()
    
    def train_models(self):
        X_train = pd.DataFrame({
            'final_text': ["login error", "payment issue", "slow performance"],
            'priority_code': [2, 1, 3],
            'desc_length': [10, 13, 16],
            'tech_flag': [1, 0, 1]
        })
        y_train = np.array(["Technical Support", "Billing", "Technical Support"])
        
        self.text_pipeline.fit(X_train['final_text'])
        text_features = self.text_pipeline.transform(X_train['final_text'])
        num_features = X_train[['priority_code', 'desc_length', 'tech_flag']].values
        X_train_final = np.hstack([text_features.toarray(), num_features])
        self.clf.fit(X_train_final, y_train)
    
    def predict_team(self, text, priority_code):
        text_feature = self.text_pipeline.transform([text])
        num_features = np.array([[priority_code, len(text), int('error' in text.lower())]])
        features = np.hstack([text_feature.toarray(), num_features])
        return self.clf.predict(features)[0]
    
    def assign_ticket(self, predicted_team):
        if self.team_load[predicted_team] < self.team_capacity[predicted_team]:
            self.team_load[predicted_team] += 1
            return predicted_team
        
        for team in sorted(self.team_load, key=lambda x: self.team_load[x]/self.team_capacity[x]):
            if self.team_load[team] < self.team_capacity[team]:
                self.team_load[team] += 1
                return team
        return "Unassigned"
    
    def save_data(self):
        self.data['tickets'].to_csv(TICKETS_FILE, index=False)
        self.data['audit_log'].to_csv(AUDIT_FILE, index=False)
        with open(KNOWLEDGE_FILE, 'w') as f:
            json.dump(self.data['knowledge_base'], f)
    
    def create_ticket(self):
        print("\n=== Create New Ticket ===")
        subject = input("Subject: ")
        description = input("Description: ")
        priority = input("Priority (Low/Medium/High/Critical): ").capitalize()
        
        if not description:
            print("Error: Description cannot be empty!")
            return
        
        new_ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        predicted_team = self.predict_team(
            f"{subject} {description}", 
            ["Low", "Medium", "High", "Critical"].index(priority)
        )
        assigned_team = self.assign_ticket(predicted_team)
        
        new_ticket = {
            'ticket_id': new_ticket_id,
            'subject': subject,
            'description': description,
            'priority': priority,
            'assigned_team': assigned_team,
            'status': 'Open',
            'created_at': pd.to_datetime(datetime.now()),
            'resolved_at': pd.NaT,
            'resolution': None
        }
        
        self.data['tickets'] = pd.concat([
            self.data['tickets'],
            pd.DataFrame([new_ticket])
        ], ignore_index=True)
        
        self.data['audit_log'] = pd.concat([
            self.data['audit_log'],
            pd.DataFrame([{
                'timestamp': datetime.now(),
                'ticket_id': new_ticket_id,
                'action': 'creation',
                'user': self.current_user,
                'details': f"Assigned to {assigned_team}"
            }])
        ])
        
        print(f"\nTicket {new_ticket_id} created and assigned to {assigned_team}")
        self.save_data()
    
    def process_tickets(self):
        print("\n=== Process Tickets ===")
        
        if self.users[self.current_user]['role'] == 'admin':
            print("Available teams:", list(self.team_capacity.keys()))
            team = input("Enter team to process tickets for: ")
            if team not in self.team_capacity:
                print("Invalid team!")
                return
        else:
            team = self.users[self.current_user]['team']
        
        team_tickets = self.data['tickets'][
            (self.data['tickets']['assigned_team'] == team) & 
            (self.data['tickets']['status'] == 'Open')
        ]
        
        if team_tickets.empty:
            print(f"No open tickets assigned to {team}!")
            return
        
        print(team_tickets[['ticket_id', 'subject', 'priority']].to_string(index=False))
        ticket_id = input("\nEnter Ticket ID to resolve: ")
        
        if ticket_id not in team_tickets['ticket_id'].values:
            print("Invalid Ticket ID!")
            return
            
        selected_ticket = self.data['tickets'][self.data['tickets']['ticket_id'] == ticket_id].iloc[0]
        print(f"\nSubject: {selected_ticket['subject']}")
        print(f"Priority: {selected_ticket['priority']}")
        print(f"Description:\n{selected_ticket['description']}")
        
        resolution = input("\nResolution notes: ")
        if not resolution.strip():
            print("Resolution notes cannot be empty!")
            return
            
        idx = self.data['tickets'].index[
            self.data['tickets']['ticket_id'] == ticket_id].tolist()[0]
        self.data['tickets'].at[idx, 'status'] = 'Resolved'
        self.data['tickets'].at[idx, 'resolved_at'] = datetime.now()
        self.data['tickets'].at[idx, 'resolution'] = resolution
        self.team_load[team] -= 1
        
        self.data['audit_log'] = pd.concat([
            self.data['audit_log'],
            pd.DataFrame([{
                'timestamp': datetime.now(),
                'ticket_id': ticket_id,
                'action': 'resolution',
                'user': self.current_user,
                'details': f"Resolution: {resolution[:100]}..."
            }])
        ])
        
        print(f"\nTicket {ticket_id} resolved!")
        self.save_data()
    
    def generate_reports(self):
        print("\n=== Reports ===")
        print("\n1. Ticket Status Summary:")
        print(self.data['tickets']['status'].value_counts().to_string())
        
        print("\n2. Team Workload:")
        workload = pd.DataFrame({
            'Team': self.team_load.keys(),
            'Assigned': self.team_load.values(),
            'Capacity': self.team_capacity.values(),
            'Utilization %': [round(self.team_load[t]/self.team_capacity[t]*100, 1) for t in self.team_load]
        })
        print(workload.to_string(index=False))
        
        resolved = self.data['tickets'][self.data['tickets']['status'] == 'Resolved']
        if not resolved.empty:
            resolved['resolution_time'] = (resolved['resolved_at'] - resolved['created_at']).dt.total_seconds()/3600
            print(f"\n3. Average Resolution Time: {resolved['resolution_time'].mean():.1f} hours")
        
        print("\n4. Recent Activity (last 5 actions):")
        print(self.data['audit_log'].tail(5).to_string(index=False))
        
        print("\n5. All Tickets:")
        print(self.data['tickets'].sort_values('created_at', ascending=False).to_string(index=False))
    
    def search_tickets(self):
        print("\n=== Search Tickets ===")
        print("1. By Ticket ID\n2. By Status\n3. By Team\n4. By Date Range")
        choice = input("Search by: ")
        
        if choice == '1':
            ticket_id = input("Ticket ID: ")
            result = self.data['tickets'][self.data['tickets']['ticket_id'] == ticket_id]
        elif choice == '2':
            status = input("Status (Open/Resolved): ").capitalize()
            result = self.data['tickets'][self.data['tickets']['status'] == status]
        elif choice == '3':
            team = input("Team name: ")
            result = self.data['tickets'][self.data['tickets']['assigned_team'] == team]
        elif choice == '4':
            start = input("Start date (YYYY-MM-DD): ")
            end = input("End date (YYYY-MM-DD): ")
            mask = (self.data['tickets']['created_at'] >= start) & (self.data['tickets']['created_at'] <= end)
            result = self.data['tickets'][mask]
        else:
            print("Invalid choice")
            return
            
        print("\nSearch Results:")
        print(result.to_string(index=False)) if not result.empty else print("No tickets found")
    
    def export_data(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data['tickets'].to_csv(f"tickets_export_{timestamp}.csv", index=False)
        self.data['audit_log'].to_csv(f"audit_export_{timestamp}.csv", index=False)
        print(f"Data exported to tickets_export_{timestamp}.csv and audit_export_{timestamp}.csv")
    
    def system_tools(self):
        print("\n=== System Tools ===")
        print("1. Check SLA Compliance\n2. Rebuild Index\n3. Backup Data")
        choice = input("Select tool: ")
        if choice == '1':
            self.check_sla()
        else:
            print("Feature coming soon!")
    
    def check_sla(self):
        now = datetime.now()
        violations = self.data['tickets'][
            (self.data['tickets']['status'] == 'Open') &
            ((now - self.data['tickets']['created_at']).dt.total_seconds() > 
             self.data['tickets']['priority'].map({'Low':72, 'Medium':24, 'High':4, 'Critical':1})*3600)
        ]
        
        if not violations.empty:
            print("\n⚠️ SLA VIOLATIONS ⚠️")
            print(violations[['ticket_id', 'priority', 'created_at']])
        else:
            print("\nNo SLA violations detected")

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    system = TicketSystem()
    
    # Login
    while not system.current_user:
        print("\n=== Login ===")
        username = input("Username: ")
        password = input("Password: ")
        
        if username in system.users and system.users[username]['password'] == password:
            system.current_user = username
            print(f"Welcome, {username}!")
        else:
            print("Invalid credentials")
    
    # Main loop
    while True:
        print("\n=== Customer Support Ticket System ===")
        print("1. Create New Ticket")
        print("2. Process Tickets")
        print("3. View Reports")
        print("4. Search Tickets")
        print("5. Export Data")
        print("6. System Tools")
        print("7. Exit")
        
        choice = input("Enter choice (1-7): ")
        
        if choice == '1':
            system.create_ticket()
        elif choice == '2':
            system.process_tickets()
        elif choice == '3':
            system.generate_reports()
        elif choice == '4':
            system.search_tickets()
        elif choice == '5':
            system.export_data()
        elif choice == '6':
            system.system_tools()
        elif choice == '7':
            system.save_data()
            print("Goodbye!")
            break
        else:
            print("Invalid choice")