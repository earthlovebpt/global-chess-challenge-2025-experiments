### Using Gated Hugging Face Models in Your Submission 🔒

The goal is to ensure your models remain secure and private while still being accessible to the evaluation servers. You can achieve this by wrapping your models as publicly visible but securely gated Hugging Face repositories.

To securely use such gated public Hugging Face models in your submissions, you must grant the `aicrowd` account access to your publicly visible but gated repository. **All repository names must include "aicrowd"** to ensure validation success.

- ✅ **Valid Example**: `team-aicrowd-my-model`
- ❌ **Invalid Example**: `team-my-model`

---

### Recommended Setup for Teams

- **Single-person teams**: Create a public, gated model under your personal Hugging Face account.
- **Multi-person teams**: Create a Hugging Face organization and manage your public gated models within this organization for better team collaboration and coordination.

**Note**: Public gated models ensure that your model is secure. Only explicitly invited accounts (such as `aicrowd`) will have access, ensuring other participants cannot view or access your submissions.

---

### Step-by-Step Guide: Creating a Public Gated Hugging Face Model

1. Log in to your [Hugging Face](https://huggingface.co/) account.
2. Click on **New Model**.
3. Enter a model name (must include "aicrowd"), set visibility to **Public**, and click **Create Model**.
4. Navigate to your model's page, click the **Settings** tab.
5. Under **Access Control**, enable **"Enable Access Requests"** to gate your model.
6. Click **Save** to apply changes.

---

### Granting Access to `aicrowd`

Follow these steps to grant instant access to the `aicrowd` account:

1. Go to your model’s settings page on Hugging Face.
2. Under **Settings**, enable **Access Requests** if not already enabled.
3. Click **Add Access**, search for **aicrowd**, select the account, and click **Grant Access**.

> **Note:** This process grants access to the `aicrowd` account **instantaneously**.
---

### Important Reminders

- Failure to explicitly grant access to the `aicrowd` account will result in submission failures.
- Ensure your repository name always includes the keyword "aicrowd".
