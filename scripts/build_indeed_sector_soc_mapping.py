"""
Build a crosswalk mapping SOC codes from OEWS to Indeed sectorName values.
"""

import csv
import openpyxl

# Load all detailed SOC codes from OEWS
wb = openpyxl.load_workbook("input/oews_national_M2024_dl.xlsx", read_only=True)
ws = wb.active
codes = []
for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i == 0:
        headers = list(row)
        occ_idx = headers.index("OCC_CODE")
        title_idx = headers.index("OCC_TITLE")
        group_idx = headers.index("O_GROUP")
        naics_idx = headers.index("NAICS")
        continue
    if row[naics_idx] == "000000" and row[group_idx] == "detailed":
        codes.append((row[occ_idx], row[title_idx]))
wb.close()

# Specific OCC_CODE → sectorName overrides (checked before prefix rules)
specific = {
    # Management (11) overrides
    "11-2011": "Marketing",           # Advertising and Promotions Managers
    "11-2021": "Marketing",           # Marketing Managers
    "11-2022": "Sales",               # Sales Managers
    "11-2032": "Media & Communications",  # Public Relations Managers
    "11-3012": "Administrative Assistance",  # Administrative Services Managers
    "11-3021": "IT Systems & Solutions",     # Computer and Information Systems Managers
    "11-3031": "Banking & Finance",   # Financial Managers
    "11-3051": "Production & Manufacturing",  # Industrial Production Managers
    "11-3071": "Logistic Support",    # Transportation, Storage, and Distribution Managers
    "11-3111": "Human Resources",     # Compensation and Benefits Managers
    "11-3121": "Human Resources",     # Human Resources Managers
    "11-3131": "Human Resources",     # Training and Development Managers
    "11-9031": "Education & Instruction",  # Education and Childcare Administrators
    "11-9032": "Education & Instruction",  # Education Administrators, K-12
    "11-9033": "Education & Instruction",  # Education Administrators, Postsecondary
    "11-9039": "Education & Instruction",  # Education Administrators, All Other
    "11-9041": "Architecture",        # Architectural and Engineering Managers
    "11-9051": "Food Preparation & Service",  # Food Service Managers
    "11-9071": "Hospitality & Tourism",  # Gambling Managers
    "11-9072": "Arts & Entertainment",   # Entertainment and Recreation Managers
    "11-9081": "Hospitality & Tourism",  # Lodging Managers
    "11-9111": "Management",          # Medical and Health Services Managers
    "11-9121": "Scientific Research & Development",  # Natural Sciences Managers
    "11-9141": "Sales",               # Property, Real Estate Managers
    "11-9151": "Community & Social Service",  # Social and Community Service Managers
    "11-9161": "Security & Public Safety",    # Emergency Management Directors
    "11-9171": "Personal Care & Home Health", # Funeral Home Managers
    "11-9179": "Personal Care & Home Health", # Personal Service Managers

    # Business and Financial Operations (13) overrides
    "13-1011": "Arts & Entertainment",   # Agents of Artists, Performers, Athletes
    "13-1031": "Banking & Finance",      # Claims Adjusters
    "13-1032": "Banking & Finance",      # Insurance Appraisers
    "13-1041": "Management",             # Compliance Officers
    "13-1051": "Accounting",             # Cost Estimators
    "13-1071": "Human Resources",        # Human Resources Specialists
    "13-1074": "Management",             # Farm Labor Contractors
    "13-1075": "Human Resources",        # Labor Relations Specialists
    "13-1081": "Logistic Support",       # Logisticians
    "13-1082": "Project Management",     # Project Management Specialists
    "13-1111": "Management",             # Management Analysts
    "13-1121": "Hospitality & Tourism",  # Meeting, Convention, Event Planners
    "13-1131": "Sales",                  # Fundraisers
    "13-1141": "Human Resources",        # Compensation, Benefits Specialists
    "13-1151": "Human Resources",        # Training and Development Specialists
    "13-1161": "Marketing",              # Market Research Analysts
    "13-1199": "Management",             # Business Operations Specialists
    "13-1020": "Management",             # Buyers and Purchasing Agents
    "13-2011": "Accounting",             # Accountants and Auditors
    "13-2020": "Banking & Finance",      # Property Appraisers
    "13-2031": "Accounting",             # Budget Analysts
    "13-2041": "Banking & Finance",      # Credit Analysts
    "13-2051": "Banking & Finance",      # Financial and Investment Analysts
    "13-2052": "Banking & Finance",      # Personal Financial Advisors
    "13-2053": "Banking & Finance",      # Insurance Underwriters
    "13-2054": "Banking & Finance",      # Financial Risk Specialists
    "13-2061": "Banking & Finance",      # Financial Examiners
    "13-2071": "Banking & Finance",      # Credit Counselors
    "13-2072": "Banking & Finance",      # Loan Officers
    "13-2081": "Accounting",             # Tax Examiners
    "13-2082": "Accounting",             # Tax Preparers
    "13-2099": "Banking & Finance",      # Financial Specialists, All Other

    # Computer and Mathematical (15) overrides
    "15-1211": "IT Systems & Solutions",   # Computer Systems Analysts
    "15-1212": "IT Systems & Solutions",   # Information Security Analysts
    "15-1221": "Software Development",     # Computer and Information Research Scientists
    "15-1231": "IT Systems & Solutions",   # Computer Network Support Specialists
    "15-1232": "IT Systems & Solutions",   # Computer User Support Specialists
    "15-1241": "IT Systems & Solutions",   # Computer Network Architects
    "15-1242": "IT Systems & Solutions",   # Database Administrators
    "15-1243": "IT Systems & Solutions",   # Database Architects
    "15-1244": "IT Systems & Solutions",   # Network and Computer Systems Administrators
    "15-1251": "Software Development",     # Computer Programmers
    "15-1252": "Software Development",     # Software Developers
    "15-1253": "Software Development",     # Software QA Analysts and Testers
    "15-1254": "Software Development",     # Web Developers
    "15-1255": "Software Development",     # Web and Digital Interface Designers
    "15-1299": "IT Systems & Solutions",   # Computer Occupations, All Other
    "15-2011": "Data & Analytics",         # Actuaries
    "15-2021": "Data & Analytics",         # Mathematicians
    "15-2031": "Data & Analytics",         # Operations Research Analysts
    "15-2041": "Data & Analytics",         # Statisticians
    "15-2051": "Data & Analytics",         # Data Scientists
    "15-2099": "Data & Analytics",         # Mathematical Science, All Other

    # Architecture and Engineering (17) overrides
    "17-1011": "Architecture",             # Architects
    "17-1012": "Architecture",             # Landscape Architects
    "17-1021": "Architecture",             # Cartographers
    "17-1022": "Architecture",             # Surveyors
    "17-2051": "Civil Engineering",        # Civil Engineers
    "17-2061": "Electrical Engineering",   # Computer Hardware Engineers
    "17-2071": "Electrical Engineering",   # Electrical Engineers
    "17-2072": "Electrical Engineering",   # Electronics Engineers
    "17-2081": "Civil Engineering",        # Environmental Engineers
    "17-2112": "Industrial Engineering",   # Industrial Engineers
    "17-3011": "Architecture",             # Architectural and Civil Drafters
    "17-3012": "Electrical Engineering",   # Electrical and Electronics Drafters
    "17-3019": "Architecture",             # Drafters, All Other
    "17-3022": "Civil Engineering",        # Civil Engineering Technicians
    "17-3023": "Electrical Engineering",   # Electrical/Electronic Eng Technicians
    "17-3024": "Electrical Engineering",   # Electro-Mechanical Technicians
    "17-3025": "Civil Engineering",        # Environmental Engineering Technicians
    "17-3026": "Industrial Engineering",   # Industrial Engineering Technicians
    "17-3031": "Architecture",             # Surveying and Mapping Technicians

    # Life, Physical, and Social Science (19) overrides
    "19-3033": "Therapy",                  # Clinical and Counseling Psychologists
    "19-3034": "Education & Instruction",  # School Psychologists
    "19-5011": "Security & Public Safety", # Occupational Health and Safety Specialists
    "19-5012": "Security & Public Safety", # Occupational Health and Safety Technicians

    # Community and Social Service (21) overrides
    "21-1012": "Education & Instruction",  # Educational Counselors and Advisors
    "21-1013": "Therapy",                  # Marriage and Family Therapists
    "21-1015": "Therapy",                  # Rehabilitation Counselors
    "21-1018": "Therapy",                  # Substance Abuse/Mental Health Counselors
    "21-1092": "Security & Public Safety", # Probation Officers

    # Healthcare Practitioners (29) overrides
    "29-1011": "Therapy",                 # Chiropractors
    "29-1021": "Dental",                  # Dentists, General
    "29-1022": "Dental",                  # Oral and Maxillofacial Surgeons
    "29-1023": "Dental",                  # Orthodontists
    "29-1024": "Dental",                  # Prosthodontists
    "29-1029": "Dental",                  # Dentists, All Other Specialists
    "29-1031": "Therapy",                 # Dietitians and Nutritionists
    "29-1041": "Physicians & Surgeons",   # Optometrists
    "29-1051": "Pharmacy",               # Pharmacists
    "29-1071": "Physicians & Surgeons",   # Physician Assistants
    "29-1081": "Physicians & Surgeons",   # Podiatrists
    "29-1122": "Therapy",                # Occupational Therapists
    "29-1123": "Therapy",                # Physical Therapists
    "29-1124": "Therapy",                # Radiation Therapists
    "29-1125": "Therapy",                # Recreational Therapists
    "29-1126": "Therapy",                # Respiratory Therapists
    "29-1127": "Therapy",                # Speech-Language Pathologists
    "29-1128": "Therapy",                # Exercise Physiologists
    "29-1129": "Therapy",                # Therapists, All Other
    "29-1131": "Physicians & Surgeons",   # Veterinarians
    "29-1141": "Nursing",                # Registered Nurses
    "29-1151": "Nursing",                # Nurse Anesthetists
    "29-1161": "Nursing",                # Nurse Midwives
    "29-1171": "Nursing",                # Nurse Practitioners
    "29-1181": "Therapy",                # Audiologists
    "29-1291": "Therapy",                # Acupuncturists
    "29-1292": "Dental",                 # Dental Hygienists
    "29-2052": "Pharmacy",              # Pharmacy Technicians
    "29-2061": "Nursing",               # Licensed Practical/Vocational Nurses
    "29-2072": "Medical Information",    # Medical Records Specialists
    "29-9021": "Medical Information",    # Health Information Technologists
    "29-9091": "Sports",                 # Athletic Trainers
    "29-9092": "Medical Information",    # Genetic Counselors

    # Healthcare Support (31) overrides
    "31-1120": "Personal Care & Home Health",  # Home Health and Personal Care Aides
    "31-1131": "Nursing",               # Nursing Assistants
    "31-1132": "Nursing",               # Orderlies
    "31-1133": "Nursing",               # Psychiatric Aides
    "31-2011": "Therapy",               # Occupational Therapy Assistants
    "31-2012": "Therapy",               # Occupational Therapy Aides
    "31-2021": "Therapy",               # Physical Therapist Assistants
    "31-2022": "Therapy",               # Physical Therapist Aides
    "31-9011": "Therapy",               # Massage Therapists
    "31-9091": "Dental",                # Dental Assistants
    "31-9092": "Medical Technician",    # Medical Assistants
    "31-9093": "Medical Technician",    # Medical Equipment Preparers
    "31-9094": "Medical Information",   # Medical Transcriptionists
    "31-9095": "Pharmacy",              # Pharmacy Aides
    "31-9096": "Medical Technician",    # Veterinary Assistants
    "31-9097": "Medical Technician",    # Phlebotomists
    "31-9099": "Personal Care & Home Health",  # Healthcare Support Workers, All Other

    # Personal Care and Service (39) overrides
    "39-1013": "Hospitality & Tourism",    # Supervisors of Gambling Services
    "39-1014": "Arts & Entertainment",     # Supervisors of Entertainment/Recreation
    "39-1022": "Personal Care & Home Health",
    "39-2011": "Personal Care & Home Health",  # Animal Trainers
    "39-2021": "Personal Care & Home Health",  # Animal Caretakers
    "39-3011": "Hospitality & Tourism",    # Gambling Dealers
    "39-3012": "Hospitality & Tourism",    # Gambling and Sports Book Writers
    "39-3019": "Hospitality & Tourism",    # Gambling Service Workers
    "39-3021": "Arts & Entertainment",     # Motion Picture Projectionists
    "39-3031": "Arts & Entertainment",     # Ushers, Lobby Attendants
    "39-3091": "Hospitality & Tourism",    # Amusement and Recreation Attendants
    "39-3092": "Arts & Entertainment",     # Costume Attendants
    "39-3093": "Hospitality & Tourism",    # Locker Room Attendants
    "39-3099": "Arts & Entertainment",     # Entertainment Attendants, All Other
    "39-4011": "Personal Care & Home Health",  # Embalmers
    "39-4012": "Personal Care & Home Health",  # Crematory Operators
    "39-4021": "Personal Care & Home Health",  # Funeral Attendants
    "39-4031": "Personal Care & Home Health",  # Morticians
    "39-5011": "Personal Care & Home Health",  # Barbers
    "39-5012": "Personal Care & Home Health",  # Hairdressers
    "39-5091": "Personal Care & Home Health",  # Makeup Artists
    "39-5092": "Personal Care & Home Health",  # Manicurists
    "39-5093": "Personal Care & Home Health",  # Shampooers
    "39-5094": "Personal Care & Home Health",  # Skincare Specialists
    "39-6011": "Hospitality & Tourism",    # Baggage Porters
    "39-6012": "Hospitality & Tourism",    # Concierges
    "39-7010": "Hospitality & Tourism",    # Tour and Travel Guides
    "39-9011": "Personal Care & Home Health",  # Childcare Workers
    "39-9031": "Sports",                   # Exercise Trainers
    "39-9032": "Hospitality & Tourism",    # Recreation Workers
    "39-9041": "Community & Social Service",   # Residential Advisors
    "39-9099": "Personal Care & Home Health",  # Personal Care Workers, All Other

    # Sales (41) overrides
    "41-1011": "Retail",               # Supervisors of Retail Sales
    "41-1012": "Sales",                # Supervisors of Non-Retail Sales
    "41-2011": "Retail",               # Cashiers
    "41-2012": "Hospitality & Tourism",  # Gambling Change Persons
    "41-2021": "Retail",               # Counter and Rental Clerks
    "41-2022": "Retail",               # Parts Salespersons
    "41-2031": "Retail",               # Retail Salespersons
    "41-3031": "Banking & Finance",    # Securities/Commodities Sales
    "41-3041": "Hospitality & Tourism",  # Travel Agents
    "41-9011": "Marketing",            # Demonstrators and Product Promoters
    "41-9012": "Marketing",            # Models

    # Office and Administrative Support (43) overrides
    "43-3011": "Accounting",            # Bill and Account Collectors
    "43-3021": "Accounting",            # Billing and Posting Clerks
    "43-3031": "Accounting",            # Bookkeeping, Accounting Clerks
    "43-3041": "Hospitality & Tourism", # Gambling Cage Workers
    "43-3051": "Accounting",            # Payroll and Timekeeping Clerks
    "43-3071": "Banking & Finance",     # Tellers
    "43-3099": "Banking & Finance",     # Financial Clerks, All Other
    "43-4011": "Banking & Finance",     # Brokerage Clerks
    "43-4031": "Legal",                 # Court, Municipal Clerks
    "43-4041": "Banking & Finance",     # Credit Authorizers
    "43-4051": "Customer Service",      # Customer Service Representatives
    "43-4081": "Hospitality & Tourism", # Hotel Desk Clerks
    "43-4121": "Education & Instruction",  # Library Assistants
    "43-4131": "Banking & Finance",     # Loan Interviewers and Clerks
    "43-4141": "Banking & Finance",     # New Accounts Clerks
    "43-4161": "Human Resources",       # Human Resources Assistants
    "43-4181": "Hospitality & Tourism", # Reservation and Ticket Agents
    "43-5011": "Logistic Support",      # Cargo and Freight Agents
    "43-5021": "Logistic Support",      # Couriers and Messengers
    "43-5031": "Security & Public Safety",  # Public Safety Telecommunicators
    "43-5032": "Logistic Support",      # Dispatchers
    "43-5051": "Logistic Support",      # Postal Service Clerks
    "43-5052": "Logistic Support",      # Postal Service Mail Carriers
    "43-5053": "Logistic Support",      # Postal Service Mail Sorters
    "43-5061": "Production & Manufacturing",  # Production/Planning Clerks
    "43-5071": "Loading & Stocking",    # Shipping, Receiving, Inventory Clerks
    "43-5111": "Loading & Stocking",    # Weighers, Measurers
    "43-6012": "Legal",                 # Legal Secretaries
    "43-9041": "Banking & Finance",     # Insurance Claims Processing
    "43-9111": "Data & Analytics",      # Statistical Assistants

    # Transportation and Material Moving (53) overrides
    "53-2031": "Hospitality & Tourism",  # Flight Attendants
    "53-3031": "Driving",               # Driver/Sales Workers
    "53-3032": "Driving",               # Heavy Truck Drivers
    "53-3033": "Driving",               # Light Truck Drivers
    "53-3051": "Driving",               # Bus Drivers, School
    "53-3052": "Driving",               # Bus Drivers, Transit
    "53-3053": "Driving",               # Shuttle Drivers
    "53-3054": "Driving",               # Taxi Drivers
    "53-3099": "Driving",               # Motor Vehicle Operators, All Other
    "53-3011": "Driving",               # Ambulance Drivers
    "53-7051": "Loading & Stocking",    # Industrial Truck Operators
    "53-7061": "Cleaning & Sanitation", # Cleaners of Vehicles and Equipment
    "53-7062": "Loading & Stocking",    # Laborers, Freight, Stock Movers
    "53-7063": "Loading & Stocking",    # Machine Feeders and Offbearers
    "53-7064": "Loading & Stocking",    # Packers and Packagers
    "53-7065": "Loading & Stocking",    # Stockers and Order Fillers
    "53-7081": "Cleaning & Sanitation", # Refuse and Recyclable Material Collectors
    "53-7121": "Loading & Stocking",    # Tank Car, Truck, Ship Loaders
    "53-7199": "Loading & Stocking",    # Material Moving Workers, All Other
}

# Prefix-based defaults (2-digit SOC major group)
prefix_defaults = {
    "11": "Management",
    "13": "Banking & Finance",
    "15": "IT Systems & Solutions",
    "17": "Industrial Engineering",      # default for engineering codes not overridden
    "19": "Scientific Research & Development",
    "21": "Community & Social Service",
    "23": "Legal",
    "25": "Education & Instruction",
    "27": "Arts & Entertainment",        # will override media/communications below
    "29": "Medical Technician",          # default for healthcare practitioners
    "31": "Medical Technician",          # default for healthcare support
    "33": "Security & Public Safety",
    "35": "Food Preparation & Service",
    "37": "Cleaning & Sanitation",
    "39": "Personal Care & Home Health",
    "41": "Sales",
    "43": "Administrative Assistance",
    "45": "Production & Manufacturing",  # farming - no direct match
    "47": "Installation & Maintenance",  # construction - closest match
    "49": "Installation & Maintenance",
    "51": "Production & Manufacturing",
    "53": "Logistic Support",            # default for transportation
}

# Sub-prefix overrides (3-4 digit) applied after major group default
sub_prefix = {
    "27-3": "Media & Communications",   # Media and Communication Workers
    "27-4": "Media & Communications",   # Media and Communication Equipment Workers
    "29-12": "Physicians & Surgeons",    # All physicians and surgeons
    "27-202": "Sports",                  # Athletes, Coaches, Umpires
}

def get_sector(occ_code):
    # Check specific override first
    if occ_code in specific:
        return specific[occ_code]
    # Check sub-prefix overrides (longest match first)
    for prefix_len in [5, 4, 3]:
        prefix = occ_code[:prefix_len]
        if prefix in sub_prefix:
            return sub_prefix[prefix]
    # Fall back to 2-digit major group
    major = occ_code[:2]
    return prefix_defaults.get(major, "Management")

# Build and write the mapping
with open("input/indeed-sector-soc-mapping.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["OCC_CODE", "OCC_TITLE", "sectorName"])
    for occ_code, occ_title in codes:
        sector = get_sector(occ_code)
        writer.writerow([occ_code, occ_title, sector])

# Print summary
from collections import Counter
sectors = Counter(get_sector(c) for c, t in codes)
print(f"Mapped {len(codes)} SOC codes to {len(sectors)} Indeed sectors:")
for sector, count in sorted(sectors.items(), key=lambda x: -x[1]):
    print(f"  {sector}: {count}")
