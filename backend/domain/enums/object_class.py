"""
Object Class Enum
==================
Defines the types of objects the detector can identify
in disaster/flood imagery, along with their risk weights.
"""

from enum import Enum


class ObjectClass(str, Enum):
    """
    Detectable object classes relevant to disaster assessment.

    Maps to COCO dataset class IDs used by YOLOv8.
    """

    PERSON = "person"
    BICYCLE = "bicycle"
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    TRUCK = "truck"
    BOAT = "boat"

    @property
    def coco_id(self) -> int:
        """The COCO dataset class ID for this object."""
        return {
            ObjectClass.PERSON: 0,
            ObjectClass.BICYCLE: 1,
            ObjectClass.CAR: 2,
            ObjectClass.MOTORCYCLE: 3,
            ObjectClass.BUS: 5,
            ObjectClass.TRUCK: 7,
            ObjectClass.BOAT: 8,
        }[self]

    @property
    def risk_weight(self) -> float:
        """
        Risk weight for damage assessment scoring.

        Higher weight = more urgent / dangerous situation.
        People in flood zones are weighted highest.
        """
        return {
            ObjectClass.PERSON: 4.0,
            ObjectClass.BUS: 3.0,
            ObjectClass.TRUCK: 2.5,
            ObjectClass.CAR: 2.0,
            ObjectClass.BOAT: 1.5,
            ObjectClass.MOTORCYCLE: 1.5,
            ObjectClass.BICYCLE: 1.0,
        }[self]

    @property
    def is_vehicle(self) -> bool:
        """Whether this object is a vehicle type."""
        return self in (
            ObjectClass.CAR, ObjectClass.TRUCK, ObjectClass.BUS,
            ObjectClass.MOTORCYCLE, ObjectClass.BICYCLE, ObjectClass.BOAT,
        )

    @property
    def is_person(self) -> bool:
        """Whether this object represents a human."""
        return self == ObjectClass.PERSON

    @classmethod
    def from_coco_id(cls, coco_id: int) -> "ObjectClass | None":
        """Look up an ObjectClass by its COCO ID. Returns None if not a target class."""
        for obj_cls in cls:
            if obj_cls.coco_id == coco_id:
                return obj_cls
        return None

    @classmethod
    def target_coco_ids(cls) -> set[int]:
        """Return the set of all COCO IDs we care about (for filtering YOLO output)."""
        return {obj_cls.coco_id for obj_cls in cls}
