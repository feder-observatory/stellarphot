from dataclasses import dataclass
import re

__all__ = ["TessSubmission"]

TIC_regex = re.compile(r"[tT][iI][cC][^\d]?(?P<star>\d+)(?P<planet>\.\d\d)?")


@dataclass
class TessSubmission:
    telescope_code: str
    filter: str
    utc_start: int
    tic_id: int
    planet_number: int

    @classmethod
    def from_header(cls, header, telescope_code="", planet=0):
        # Set some default dummy values

        tic_id = 0
        filter = ""
        fails = {}
        try:
            dateobs = header['date-obs']
        except KeyError:
            fails["utc_start"] = "UTC date of first image"
        else:
            dateobs = dateobs.split("T")[0].replace("-", "")

        try:
            filter = header['filter']
        except KeyError:
            fails["filter"] = ("filter/passband")

        try:
            obj = header['object']
        except KeyError:
            fails['tic_id'] = "TIC ID number"
        else:
            result = TIC_regex.match(obj)
            if result:
                tic_id = int(result.group("star"))
                # Explicit argument overrules the header
                if result.group("planet") and not planet:
                    # Drop the leading period from the match
                    planet = int(result.group("planet")[1:])
            else:
                # No star from the object after all
                fails['tic_id'] = "TIC ID number"

        fail_msg = ""
        fail = []
        for k, v in fails.items():
            fail.append(f"Unable to determine {k}, {v}, from header.")

        fail = "\n".join(fail)

        if fail:
            raise ValueError(fail)

        return cls(utc_start=dateobs,
                   filter=filter,
                   telescope_code=telescope_code,
                   tic_id=tic_id,
                   planet_number=planet)

    def _valid_tele_code(self):
        return len(self.telescope_code) > 0

    def _valid_planet(self):
        return self.planet_number > 0

    def _valid_tic_num(self):
        return self.tic_id < 10_000_000_000

    def _valid(self):
        """
        Check whether the information so far is valid, meaning:
         + Telescope code is not the empty string
         + Planet number is not zero
         + TIC ID is not more than 10 digits
        """
        valid = (
            self._valid_tele_code() and
            self._valid_planet() and
            self._valid_tic_num()
        )
        return valid

    @property
    def base_name(self):
        if self._valid():
            pieces = [
                f"TIC{self.tic_id}-{self.planet_number:02d}",
                self.utc_start,
                self.telescope_code,
                self.filter
            ]
            return "_".join(pieces)

    @property
    def seeing_profile(self):
        return self.base_name + "_seeing-profile"

    def invalid_parts(self):
        if self._valid():
            return

        if not self._valid_tele_code():
            print(f"Telescope code {self.telescope_code} is not valid")

        if not self._valid_planet():
            print(f"Planet number {self.planet_number} is not valid")

        if not self._valid_tic_num():
            print(f"TIC ID {self.tic_id} is not valid.")
