import sqlite3
import numpy as np
import boardlib
import pickle

class KilterClimbGenData:

  def __init__(self, angle, grade, string, nomatch):
    self.angle = angle
    self.grade = grade
    self.string = string
    self.nomatch = nomatch


# TrainGenerator.py's KilterDataSet pads every sequence into a fixed 153-slot
# tensor (150 hold tokens + 3 conditioning tokens). A handful of junk climbs
# (e.g. "every hold lit up" test climbs) blow way past that, which crashes
# DataLoader's collate when it tries to stack them against normal climbs.
MAX_HOLD_TOKENS = 150


def hold_token_count(framestring):
    count = 0
    token = ""
    for char in framestring:
        if char == 'p' or char == 'r':
            if token != "":
                count += 1
            token = ""
        else:
            token = token + char
    if token != "":
        count += 1
    return count


conn = sqlite3.connect("climbs.db")
cursor = conn.cursor()

# Get layout_id for Kilter Board Original
cursor.execute("SELECT id FROM layouts WHERE product_id = (SELECT id FROM products WHERE name = 'Kilter Board Original')")
layout_id = cursor.fetchone()[0]

# Get placements and holes for THIS layout only
cursor.execute(
    "SELECT p.id, h.x, h.y FROM placements p JOIN holes h ON p.hole_id = h.id WHERE p.layout_id = ?;",
    (layout_id,)
)
coord_map = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

cursor.execute("SELECT id, name FROM placement_roles")
color = {id: name for id, name in cursor.fetchall()}

cursor.execute("SELECT difficulty, boulder_name FROM difficulty_grades")
name_to_grade = {boulder_name.split('/')[1]: difficulty for difficulty, boulder_name in cursor.fetchall()}




cursor.execute("""SELECT
    cs.angle,
    dg.difficulty,
    c.frames,
    c.is_nomatch
    FROM climbs c
    JOIN climb_stats cs ON c.uuid = cs.climb_uuid
    JOIN layouts l ON c.layout_id = l.id
    JOIN products p ON l.product_id = p.id
    JOIN difficulty_grades dg ON dg.difficulty = ROUND(cs.display_difficulty)
    WHERE p.name = 'Kilter Board Original' AND c.frames_count = 1 AND cs.ascensionist_count >= 5;""")


climbs = []

for row in cursor:
  angle, grade, framestring, nomatch = row
  if hold_token_count(framestring) > MAX_HOLD_TOKENS:
    continue
  climb = KilterClimbGenData(angle,grade,framestring,nomatch)
  climbs.append(climb)


with open ("KilterClimbsGenerationData.pkl", "wb") as file:
  pickle.dump(climbs, file)

with open ("KilterMap.pkl", "wb") as file:
  pickle.dump(coord_map, file)
  pickle.dump(color,file)
  pickle.dump(name_to_grade,file)


conn.close()
