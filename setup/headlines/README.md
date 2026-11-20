# UK headlines dataset

`hear_uk_headlines.py` builds `uk_headlines.json`, the dataset used in section 7 of the course. It samples the lead headline of each UK outlet from [The Hear](https://www.thehear.org) archive every 6 hours, from September 2024 to the day it runs, and caches the raw responses in `hear_cache/` so it can resume.

The course copy was collected in October 2026 and lives in `$PROJECTDIR/course/headlines/`. To rebuild it, run this script from that folder on a node with internet access (it uses only the Python standard library):

```bash
cd $PROJECTDIR/course/headlines
python $PROJECTDIR/hpc-for-ai-2/setup/headlines/hear_uk_headlines.py   # every 6 hours, full archive
```
