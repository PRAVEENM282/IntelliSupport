import logging
import re
from typing import Optional

from pydantic import BaseModel


logger = logging.getLogger(__name__)
DOC_ID_PATTERN = re.compile(r"^doc_\d{3}$")


class Document(BaseModel):
    doc_id: str
    title: str
    content: str
    source_url: Optional[str] = None
    metadata: dict = {}


def _seed_content(
    title: str,
    focus: str,
    capabilities: str,
    procedure: str,
    policy: str,
    troubleshooting: str,
) -> str:
    return (
        f"{title} is part of Nexora's official customer support knowledge base for project "
        f"administrators, workspace owners, and everyday contributors. This article focuses on "
        f"{focus}. Nexora is designed for B2B project management teams that coordinate tasks, "
        f"deadlines, files, approvals, and customer-facing delivery work in one shared workspace. "
        f"The guidance below is approved for support responses and should be used when customers "
        f"ask about the behavior, configuration, limitations, or recommended operating practices "
        f"for this area of the product.\n\n"
        f"Core capabilities include {capabilities}. Customers should start by confirming that they "
        f"are signed in to the correct Nexora workspace, because many settings are scoped to a "
        f"specific workspace rather than to a personal profile. Workspace owners and admins can "
        f"usually make configuration changes from the Settings area, while members may only see "
        f"options that match their assigned role. Nexora records important administrative changes "
        f"in the activity log so support teams can help audit what changed, when it changed, and "
        f"which user performed the action.\n\n"
        f"Recommended procedure: {procedure}. After making a change, customers should refresh the "
        f"project or workspace view and verify the result with a small test before rolling it out "
        f"to a larger team. For example, an admin can create a sample project, invite a limited "
        f"group of users, trigger a notification, export a test file, or send a test integration "
        f"event depending on the feature being configured. This reduces the risk of disrupting "
        f"active work and gives the customer a clear troubleshooting point if something does not "
        f"behave as expected.\n\n"
        f"Important policy and limits: {policy}. Nexora support can explain documented behavior "
        f"and help customers review settings, but support cannot bypass security controls, change "
        f"billing ownership without verification, expose private project data, or guarantee that "
        f"third-party services will accept every request. If a customer asks for information that "
        f"is not covered in this article, the correct response is to say that the available "
        f"documentation does not contain that information and ask a clarifying question.\n\n"
        f"Troubleshooting guidance: {troubleshooting}. Customers should capture the workspace name, "
        f"project name, affected user email, approximate time of the issue, and any visible error "
        f"message before contacting support. For repeatable problems, Nexora recommends testing in "
        f"a single project with one administrator and one member account. This makes it easier to "
        f"separate product behavior from permissions, browser cache, expired sessions, integration "
        f"tokens, plan limits, or organization-level security settings."
    )

# test seed data
SEED_DOCUMENTS: list[dict] = [
    {
        "doc_id": "doc_001",
        "title": "Getting Started with Nexora",
        "source_url": "https://support.nexora.example/docs/getting-started",
        "metadata": {"category": "onboarding"},
        "content": _seed_content(
            "Getting Started with Nexora",
            "creating a workspace, setting up the first project, inviting teammates, and learning the main navigation",
            "workspace creation, project dashboards, task lists, timeline views, file attachments, comments, mentions, saved filters, and guided onboarding checklists",
            "create the workspace, set the company name and time zone, add one project, choose a template or blank workflow, invite a small pilot team, assign roles, and review the dashboard together before inviting the whole organization",
            "only workspace owners can transfer ownership, delete the workspace, or configure organization-wide defaults; project admins can manage their own projects but cannot change billing or global security settings",
            "if a new user cannot see a project, confirm they accepted the invitation, joined the correct workspace, and were added to the project rather than only to the organization directory",
        ),
    },
    {
        "doc_id": "doc_002",
        "title": "Managing Team Members and Permissions",
        "source_url": "https://support.nexora.example/docs/team-permissions",
        "metadata": {"category": "account_management"},
        "content": _seed_content(
            "Managing Team Members and Permissions",
            "inviting members, assigning roles, removing access, and controlling what users can view or edit",
            "workspace invitations, role-based access control, guest access, project-level permissions, admin roles, member roles, read-only reviewers, group assignment, and audit history for permission changes",
            "open Workspace Settings, choose Members, invite the user by email, select a role, add the user to the relevant projects, and send the invitation; to change access later, edit the member record and save the new role or remove them from selected projects",
            "removing a member immediately blocks access to private projects, but historical comments and task ownership remain visible for audit continuity; only owners can promote another owner or remove the final owner account",
            "if a member cannot access a project, check whether they are only invited at the workspace level, whether their invitation is still pending, whether single sign-on created a different account, and whether a project admin restricted the project to a smaller group",
        ),
    },
    {
        "doc_id": "doc_003",
        "title": "Billing and Subscription Plans",
        "source_url": "https://support.nexora.example/docs/billing-plans",
        "metadata": {"category": "billing"},
        "content": _seed_content(
            "Billing and Subscription Plans",
            "subscription upgrades, plan limits, invoices, cancellation, renewals, refunds, and payment methods",
            "monthly and annual plans, seat-based billing, invoice history, downloadable receipts, plan upgrades, downgrade scheduling, tax details, purchase order notes, billing owner transfer, and cancellation workflows",
            "open Billing Settings, confirm the billing owner, review the current plan, choose Upgrade, Downgrade, or Cancel, verify the seat count, enter payment details if needed, and save the change after reviewing the renewal date and prorated charges",
            "plan upgrades apply immediately and may create prorated charges, while downgrades and cancellations usually take effect at the end of the paid billing period; deleting projects does not automatically reduce the subscribed seat count",
            "if a payment fails, verify the card, billing address, tax information, bank authorization, and invoice email; if a customer cancels, their workspace remains available until the end of the current term unless the account is already past due",
        ),
    },
    {
        "doc_id": "doc_004",
        "title": "Integrations: Slack, GitHub, and Jira",
        "source_url": "https://support.nexora.example/docs/integrations",
        "metadata": {"category": "integration"},
        "content": _seed_content(
            "Integrations: Slack, GitHub, and Jira",
            "connecting Nexora to Slack, GitHub, and Jira so teams can receive updates and synchronize work across tools",
            "Slack channel notifications, GitHub repository linking, pull request activity, Jira issue syncing, OAuth authorization, integration permission checks, event filters, and per-project integration settings",
            "open Project Settings, choose Integrations, select Slack, GitHub, or Jira, authorize the third-party service, pick the workspace, repository, channel, or project to connect, configure event filters, and send a test event before enabling notifications for the whole team",
            "third-party integrations require an active account in the external service, and Nexora cannot connect resources the authorizing user cannot access; revoking OAuth access in Slack, GitHub, or Jira will stop Nexora events until the connection is authorized again",
            "if Slack messages do not appear, confirm the channel is selected and the Nexora app is allowed in that workspace; for GitHub or Jira sync issues, refresh the OAuth token, verify repository or project permissions, and check whether event filters are excluding the expected updates",
        ),
    },
    {
        "doc_id": "doc_005",
        "title": "Project Templates and Workflows",
        "source_url": "https://support.nexora.example/docs/templates-workflows",
        "metadata": {"category": "feature_request"},
        "content": _seed_content(
            "Project Templates and Workflows",
            "using built-in templates, creating custom templates, and standardizing workflows for recurring projects",
            "template galleries, custom project templates, task stages, workflow statuses, approval steps, required fields, project duplication, milestone presets, automation rules, and default assignee patterns",
            "open Templates, choose a built-in template or create one from an existing project, review task groups and workflow stages, remove customer-specific data, set default owners or due-date offsets, and publish the template for the workspace",
            "custom templates are available on team and business plans; template changes do not rewrite projects that were already created, and only admins can publish shared templates for everyone in the workspace",
            "if a template does not appear when creating a project, confirm it is published, the user has access to the workspace template library, the plan supports custom templates, and the template has not been archived by another admin",
        ),
    },
    {
        "doc_id": "doc_006",
        "title": "Notifications and Alert Settings",
        "source_url": "https://support.nexora.example/docs/notifications",
        "metadata": {"category": "notifications"},
        "content": _seed_content(
            "Notifications and Alert Settings",
            "configuring email, in-app, Slack, and digest notifications so users receive useful alerts without unnecessary noise",
            "personal notification preferences, project notification defaults, mentions, task assignment alerts, due-date reminders, daily digests, weekly summaries, Slack routing, quiet hours, and unsubscribe controls",
            "open Personal Settings for individual preferences or Project Settings for project defaults, choose the notification channels, enable alerts for mentions and assigned tasks, set reminder timing, and save a quiet-hours window if the user does not want alerts outside work hours",
            "workspace admins can set defaults, but individual users may override many personal alerts; security, billing, and ownership notices may still be sent even when optional project notifications are disabled",
            "if a user is missing notifications, check spam folders, email verification, quiet hours, disabled project alerts, Slack channel membership, and whether the event happened before the user was added to the project",
        ),
    },
    {
        "doc_id": "doc_007",
        "title": "Data Export and Backup",
        "source_url": "https://support.nexora.example/docs/export-backup",
        "metadata": {"category": "data_and_export"},
        "content": _seed_content(
            "Data Export and Backup",
            "exporting project data, downloading CSV files, creating backups, and understanding what information is included",
            "CSV exports, project-level exports, workspace backups, task history, comments, attachments metadata, audit events, scheduled backup requests, admin-only export controls, and retention guidance",
            "open Workspace Settings or Project Settings, choose Export Data, select CSV or full backup where available, choose the project scope, confirm the date range, start the export, and download the generated file from the export center when processing completes",
            "only workspace owners and authorized admins can export all workspace data; CSV exports include task names, status, assignee, due date, labels, and custom fields, but attached files may be provided as links or separate download packages depending on the plan",
            "if an export is delayed or missing data, confirm the requester has export permission, verify the selected date range, check whether archived projects were included, and retry with a smaller project scope before contacting support",
        ),
    },
    {
        "doc_id": "doc_008",
        "title": "Two-Factor Authentication Setup",
        "source_url": "https://support.nexora.example/docs/two-factor-authentication",
        "metadata": {"category": "account_management"},
        "content": _seed_content(
            "Two-Factor Authentication Setup",
            "enabling two-factor authentication, using authenticator apps, storing recovery codes, and recovering access after login problems",
            "2FA setup, authenticator app enrollment, time-based one-time passwords, backup recovery codes, enforced 2FA for organizations, password reset interaction, trusted devices, and account recovery verification",
            "open Account Settings, choose Security, select Enable Two-Factor Authentication, scan the QR code with an authenticator app, enter the six-digit code, save recovery codes in a secure location, and sign out and back in to verify the setup",
            "organization-enforced 2FA may require every member to enroll before accessing projects; Nexora support cannot read authentication codes or bypass two-factor authentication without completing the documented account recovery process",
            "if a user forgot a password and cannot log in, start with password reset, then use a current authenticator code or recovery code; if the device was lost, collect workspace details and begin the verified recovery process with support",
        ),
    },
    {
        "doc_id": "doc_009",
        "title": "API Access and Webhooks",
        "source_url": "https://support.nexora.example/docs/api-webhooks",
        "metadata": {"category": "technical"},
        "content": _seed_content(
            "API Access and Webhooks",
            "creating API tokens, configuring webhook endpoints, subscribing to events, and diagnosing delivery failures",
            "personal API tokens, scoped access keys, webhook endpoints, event subscriptions, signing secrets, retry behavior, delivery logs, payload examples, rate limits, token rotation, and deactivation controls",
            "open Developer Settings, create a scoped API token, copy it once, add a webhook endpoint URL, select events such as task.created or project.updated, save the signing secret, send a test event, and confirm that the receiving service records the payload",
            "API tokens inherit the access of the creating user and should be rotated if exposed; webhook delivery requires a publicly reachable HTTPS endpoint that returns a successful 2xx response within the timeout window",
            "if a webhook is not receiving events, inspect the delivery log, verify the endpoint URL, check firewalls, confirm the event subscription is enabled, validate the signing secret, and ensure the receiving server is not returning 4xx or 5xx errors",
        ),
    },
    {
        "doc_id": "doc_010",
        "title": "Troubleshooting Common Errors",
        "source_url": "https://support.nexora.example/docs/troubleshooting",
        "metadata": {"category": "technical_issue"},
        "content": _seed_content(
            "Troubleshooting Common Errors",
            "diagnosing common Nexora errors, crashes, unexpected behavior, failed saves, slow pages, and sync issues",
            "browser cache checks, session refresh, network diagnostics, error code lookup, failed save recovery, project loading issues, attachment upload troubleshooting, status page review, and escalation preparation",
            "record the exact error message, refresh the browser, test in a private window, clear cached Nexora assets, confirm network access, check the Nexora status page, reproduce the issue in one project, and gather timestamps before contacting support",
            "some errors are caused by permission restrictions, expired sessions, plan limits, unsupported file types, or third-party service outages rather than by a Nexora platform incident",
            "if the app crashes when opening a project, test another project, disable browser extensions, confirm the user has project access, capture the console error if available, and include the project URL and approximate crash time in the support ticket",
        ),
    },
]


class DocumentLoader:
    def load_from_dict(self, data: dict) -> Document:
        doc_id = data.get("doc_id")
        content = data.get("content")
        if not isinstance(doc_id, str) or not DOC_ID_PATTERN.match(doc_id):
            raise ValueError("doc_id must match pattern ^doc_\\d{3}$")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must not be empty")
        return Document(
            doc_id=doc_id,
            title=data["title"],
            content=content,
            source_url=data.get("source_url"),
            metadata=data.get("metadata") or {},
        )

    def load_batch(self, data_list: list[dict]) -> list[Document]:
        documents: list[Document] = []
        for item in data_list:
            try:
                documents.append(self.load_from_dict(item))
            except Exception as exc:
                logger.warning("Skipping invalid document: %s", exc)
        return documents

    def save_to_db(self, documents: list[Document], conn) -> int:
        from psycopg2.extras import Json

        if not documents:
            return 0

        rows_written = 0
        with conn.cursor() as cur:
            for document in documents:
                cur.execute(
                    """
                    INSERT INTO intellisupport.documents
                        (doc_id, title, source_url, content, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        updated_at = NOW()
                    """,
                    (
                        document.doc_id,
                        document.title,
                        document.source_url,
                        document.content,
                        Json(document.metadata),
                    ),
                )
                rows_written += cur.rowcount
        conn.commit()
        return rows_written


def load_seed_documents() -> list[Document]:
    return DocumentLoader().load_batch(SEED_DOCUMENTS)
