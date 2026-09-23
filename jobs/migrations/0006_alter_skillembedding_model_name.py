from django.db import migrations, models


class Migration(migrations.Migration):
    """Cosmetic only: updates SkillEmbedding.model_name's default to match
    EmbeddingService.MODEL_NAME (jobs/services/embeddings.py) after the
    MiniLM -> bge-small-en-v1.5 swap. The field is always written explicitly
    by EmbeddingService, so this default is never actually relied on - it
    just keeps the schema from looking stale if inspected directly."""

    dependencies = [
        ('jobs', '0005_jobembedding'),
    ]

    operations = [
        migrations.AlterField(
            model_name='skillembedding',
            name='model_name',
            field=models.CharField(default='BAAI/bge-small-en-v1.5', max_length=100),
        ),
    ]
