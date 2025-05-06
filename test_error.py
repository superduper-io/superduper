from superduper import superduper
from templates.agent.help import add_app_from_template

db = superduper()
add_app_from_template(db, "./agent_template_simple")
