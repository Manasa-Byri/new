# Data Folder

This folder contains CSV and other data files used by the application for insights generation.

## Current Files

### Insurance Data
- **USRW.NONX.IYM551ND.MEMBER.SWEEP.G2262V.csv**
  - 49,999 insurance member records
  - US Region (ISG - Individual & Small Groups)
  - Coverage: Medical, Dental, Vision
  - Used by: `csv_insurance_insights_service.py`

## Adding New Data Files

1. Place CSV files in this folder
2. Update the service layer to reference the new file
3. Follow the same pattern as `csv_insurance_insights_service.py`

## Best Practices

- Keep data files within the `app/data/` folder
- Use descriptive file names
- Document file structure and purpose
- Use relative paths from the service layer
- Add `.gitignore` entry for sensitive data files if needed
