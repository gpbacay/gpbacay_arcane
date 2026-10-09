# ARC 1 Lending App Business

Yes. A lending app is a promising ARC 1 application—but I would sell ARC 1 as the automation layer for licensed merchants, rather than immediately lending your own money.

## Best Business: Loan Operations API

Build a subscription service that receives customer loan applications and automates repetitive processing:

```text
Application/OCR text
        ↓
ARC 1
        ├── Extract customer details
        ├── Detect missing requirements
        ├── Classify application type
        ├── Route suspicious cases
        ├── Prioritize manual review
        └── Trigger approved workflows
```

## Examples

| Customer input | ARC 1 result |
|---|---|
| “I’m employed by ABC Corp and earn ₱35,000 monthly” | Extract employer, employment type, and income |
| “I don’t have a payslip, but I have bank statements” | `missing_income_document` |
| “My ID name is Juan Santos, but my application says John Santos” | `identity_mismatch_review` |
| “I need ₱20,000 for motorcycle repairs” | Classify loan purpose and extract amount |
| “Can I move my payment to October 30?” | Call `request_due_date_change(date="October 30")` |
| “I already paid through GCash yesterday” | Route to `payment_verification` |
| “I lost my job and cannot make this month’s payment” | Route to `hardship_assistance`, not aggressive collection |
| “Why was my application rejected?” | Route to a human or compliant explanation workflow |

ARC 1 can also classify uploaded customer documents after OCR:

- Government ID
- Payslip
- Bank statement
- Proof of billing
- Employment certificate
- Business permit
- Unsupported or unreadable document

ARC 1 itself does not read images; OCR or document-vision software would first turn the document into text.

## What ARC 1 Should Not Decide Alone

Do not let ARC 1 directly determine:

- Whether a customer deserves a loan
- The interest rate
- The maximum credit limit
- Whether a customer is fraudulent
- Whether collection action should begin

Instead, let it produce an operational recommendation:

```json
{
  "route": "manual_underwriting",
  "reason_codes": [
    "income_document_missing",
    "name_mismatch"
  ],
  "confidence": 0.87
}
```

The merchant’s documented underwriting rules, credit data, affordability calculations, and authorized reviewers should make the actual credit decision. This is also technically important: `arc1-tiny` was not validated as a credit-risk model.

## How It Earns Recurring Revenue

Sell it to existing lending and financing merchants:

- Starter: ₱2,500/month for 2,000 applications
- Growth: ₱10,000/month for 15,000 applications
- Business: ₱30,000+/month with private deployment
- Optional usage charge: ₱0.25–₱1 per processed application
- Setup and fine-tuning: ₱20,000–₱100,000 per merchant

Illustrative example:

```text
20 merchants × ₱10,000/month = ₱200,000 monthly recurring revenue
Hosting and inference             - ₱20,000
Support and operations            - ₱40,000
Illustrative remainder             ₱140,000/month
```

That is not a forecast—customer acquisition, compliance, support, accuracy, and competition determine the real result.

## Strongest Selling Points

ARC 1 gives you several commercially useful claims after you validate them on merchant data:

- Private, on-premise processing
- Very low inference cost
- Fast application routing
- Typed output instead of generated prose
- Extracted values grounded in submitted text
- Confidence thresholds and human escalation
- No customer information sent to a third-party LLM

The privacy angle may be more valuable than the model size. Merchants handle customer identity, income, and financial information and may want processing inside their own infrastructure.

## Regulatory Boundary in the Philippines

Operating the automation software is considerably simpler than operating the lending company. A merchant actually granting loans must obtain authority from the SEC; an ordinary app developer cannot simply launch an unlicensed lending operation. [The SEC explains the authority-to-operate requirement](https://appointment.sec.gov.ph/lending-companies-and-financing-companies-2/lending-companies-and-financing-companies/).

Customers must be informed when loan processing involves profiling, automated processing, automated decisions, or credit scoring. They must also be told the categories of data considered. [NPC loan-processing guidelines](https://privacy.gov.ph/wp-content/uploads/2022/02/NPC-Circular-No.-20-01.pdf).

The NPC also requires notification where automated processing becomes the sole basis for a decision that significantly affects a customer. Keeping ARC 1 as an assisted-triage system with human review reduces—but does not eliminate—the compliance burden. [NPC rules on automated decision-making](https://privacy.gov.ph/npc-circular-17-01-registration-data-processing-notifications-regarding-automated-decision-making/).

## Recommendation

Start with one narrow product: **“AI document checker and application router for small licensed merchants.”** It has recurring value, avoids risking your own capital, and fits ARC 1 much better than autonomous credit scoring.
